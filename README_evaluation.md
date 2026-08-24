# Evaluation Framework

This document describes the evaluation and hyperparameter tuning setup for all motion models.

## Position in the Pipeline

After preprocessing, evaluation consumes windowed trajectory data and produces:

1. aggregate metrics (ADE, FDE, RMSE),
2. per-file and per-month diagnostics,
3. optional prediction exports for visual inspection.

Data paths:

- input: `output/07_parquet/`
- outputs: `output/08_baseline_results/`

## System Requirements

- Python dependencies: see `requirements.txt`
- NVIDIA CUDA 8.0 or later for TiRex and the LSTM models
- 32+ GB RAM recommended (tested on 48 GB) for caching track data during tuning and evaluation
- Chronos-2 on Linux requires `export HF_HUB_DISABLE_XET=1` before running so model downloads work reliably
- **TiRex CUDA backend:** `xlstm` calls `torch.utils.cpp_extension.include_paths(cuda=True)` which was removed in PyTorch 2.11+. Patch `cuda_init.py`: replace `include_paths(cuda=True)` with `include_paths()` and hardcode `CUDA_LIB` to your CUDA lib path (e.g. `/usr/local/cuda-12.6/targets/x86_64-linux/lib`). This patch must be reapplied after `pip install --upgrade xlstm`. As a fallback if CUDA kernel compilation fails, set `backend: torch` in `configs/evaluation.yaml` under `models.tirex` to use PyTorch ops on GPU without custom kernels (3-4x slower).

## Model Layer

### `models/base_model.py`

`BaselineModel` is the shared interface for all models. The evaluator is model-agnostic and only depends on `predict(context_df, n_pred_steps)`.

### Kinematic models: `models/kinematic.py`

- `ConstantVelocityModel` (CV): averages recent velocity over a tuned context window.
- `ConstantTurnRateVelocityModel` (CTRV): same averaging, but also propagates heading and turn rate.
- `ConstantTurnRateVelocityArcModel` (CTRV Arc): CTRV variant that reconstructs position along an arc instead of straight steps.
- `HybridCVCTRVModel` (Hybrid): selects CV or CTRV per track based on a rotation-rate threshold.

CV, CTRV, and CTRV Arc share one tuning parameter: `velocity_steps` (how many recent context steps to average). The Hybrid model has four: `cv_velocity_steps`, `ctrv_velocity_steps`, `rot_steps`, and `rot_threshold`.

### Filter models: `models/filters.py`

- `KalmanFilter` (Kalman): linear constant-velocity Kalman filter. Runs a predict-correct cycle over the context window, then rolls out without further corrections.
- `CTRVExtendedKalmanFilter` (CTRV EKF): extended Kalman filter using the CTRV motion model. Also runs a predict-correct cycle, then rolls out.

Kalman tuning parameters: `q_pos`, `q_vel`, `r_pos`, `p0_pos`, `p0_vel`.
CTRV EKF tuning parameters: `q_pos`, `q_vel`, `q_rot`, `r_pos`, `p0_pos`, `p0_vel`, `p0_rot`, `init_velocity_steps`.

### Foundation models: `models/sequence/`

- `TirexLSTMModel` (TiRex): pre-trained xLSTM-based foundation model (NX-AI/TiRex). Loaded from HuggingFace, no training required. One tuning parameter: `velocity_steps` (size of the context window fed to the model).
- `Chronos2ZeroShotModel` (Chronos-2): pre-trained probabilistic transformer (Amazon Chronos-2). Fully zero-shot, no tuning of any kind. Uses the median quantile (`q=0.5`) as the point forecast and also outputs prediction intervals.

### Trained LSTM models: `models/sequence/minimal_lstm.py`

- `MinimalLSTMModel` (Minimal LSTM): LSTM trained from scratch on the training split. Encodes a fixed-length context window, then predicts the full horizon in one shot via a linear head (no autoregressive decoding).
- `MinimalLSTMDomainModel` (OHE LSTM): same architecture, but the waterway domain (river, harbour, channel, lock) is injected as a one-hot vector into the initial LSTM hidden state.

Both models use normalized displacement (`dx_norm`, `dy_norm`) as input and target.

## Evaluation Layer

### `evaluation/metrics.py`

Provides trajectory quality metrics:

- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- RMSE
- ADE per prediction step
- MIW (Mean Interval Width) from the `[q=0.1, q=0.9]` prediction interval
- Coverage of the `[q=0.1, q=0.9]` interval
- Winkler interval score

MIW, Coverage, and Winkler score apply only to models that output quantile predictions (TiRex, Chronos-2).

### `evaluation/evaluator.py`

Core responsibilities:

- resolve files by split and context,
- load and group tracks,
- call model prediction,
- reconstruct positions from predicted displacements,
- aggregate metrics.

Two execution modes are available:

1. streaming mode (file-by-file),
2. cached numpy mode for fast repeated evaluation during tuning.

Cached mode is used by Optuna to avoid repeated disk I/O per trial.

## Hyperparameter Tuning

### `evaluation/tune_hyperparams.py`

Tunes or trains all models on the **validation split**.

**Grid search (1D models):**

CV, CTRV, CTRV Arc, and TiRex LSTM each have one tuning parameter (`velocity_steps`). The tuner tests every integer from 1 to `max_velocity_steps` and picks the value with the lowest validation RMSE. Each model runs in a separate process.

**Optuna TPE (multi-parameter kinematic models):**

- Hybrid: 4 parameters, seeded with the grid-search results for CV and CTRV.
- Kalman: 5 parameters.
- CTRV EKF: 8 parameters, with early stopping.

**Neural training (LSTM models):**

Minimal LSTM and OHE LSTM are trained from scratch inside the tuning script. Optuna searches over hyperparameter configurations (hidden size, dropout, learning rate, batch size); each trial trains a model from scratch on the training split and evaluates it on the validation split. The best checkpoint is saved to the path configured in `evaluation.yaml`. These models do not read from `tuning_best_params_val.csv`.

**Chronos-2:**

No tuning: fully zero-shot.

**Tuning procedure:**

1. 1D grid-search jobs run in parallel,
2. validation tracks are cached in RAM per worker process,
3. results are collected and an intermediate `tuning_best_params_val.csv` is written,
4. Hybrid, Kalman, and CTRV EKF Optuna studies run sequentially,
5. LSTM models train sequentially after the kinematic tuning,
6. final best-params and all-trials CSVs are written.

Outputs (written to `output/08_baseline_results/runs/{run.name}/tuning/`).


Splits:

- validation is used for parameter selection and LSTM training,
- test is used only for final reporting.

## Final Evaluation

### `evaluation/run_evaluation.py`

Runs final metrics on the configured split (default: `test`) for all enabled models.

Behavior:

1. loads tuned parameters from `tuning_best_params_val.csv` (required for kinematic and TiRex models; crashes if missing: run tuning first),
2. loads all parquet files into memory once and reuses them across all models,
3. evaluates on the configured split,
4. computes per-file, per-month, and per-context (harbour/river/channel/lock) metrics,
5. optionally exports per-model prediction CSVs.

Chronos-2 does not read from `tuning_best_params_val.csv`. It runs zero-shot and outputs quantile-aware diagnostics in addition to the standard metrics.

The LSTM models load their checkpoint from the path configured in `evaluation.yaml` and also do not use `tuning_best_params_val.csv`.

On Linux, set `HF_HUB_DISABLE_XET=1` before running Chronos-2 (`export HF_HUB_DISABLE_XET=1`).

## Plotting and Reporting

### `evaluation/plot_evaluation.py`

Generates all plots from existing CSV outputs without re-running model inference.

Behavior:

1. reads diagnostics from the configured run folder,
2. generates ADE-over-horizon plots per model,
3. generates monthly ADE/FDE/RMSE plots per model,
4. generates context-wise metric and uncertainty interval plots,
5. generates pairwise RMSE violin plots for selected model pairs,
6. writes all images to the run's `plots/` folder.

Model visibility is controlled by the `plotting.models` flags in `configs/evaluation.yaml`.

## Usage Sequence

1. run `evaluation/tune_hyperparams.py` (trains LSTM models and tunes all other models on validation),
2. verify `tuning_best_params_val.csv`,
3. run `evaluation/run_evaluation.py` on test,
4. run `evaluation/plot_evaluation.py` to generate plots,
5. compare final metrics and figures.

## Central Config

All runtime settings are in `configs/evaluation.yaml`:

- run metadata (`run.name`, `run.seed`, `run.sample_pct`),
- sampling fractions for tuning and evaluation (`sampling.*`),
- data paths and split (`data.*`),
- model enable switches (`models.*`),
- per-model configuration (device, checkpoint paths, batch sizes),
- tuning parameters (`tuning.*`),
- plot visibility and step duration (`plotting.*`).

`sampling.tuning_validation_pct` subsamples the validation split during tuning.
`sampling.evaluation_pct` subsamples the evaluation split (useful for quick test runs).
All sampling is deterministic via `run.seed`.

## Output Layout

All outputs are versioned under the run name to avoid overwriting earlier results:

- base: `output/08_baseline_results/runs/{run.name}/`
- tuning outputs: `.../tuning/`
- evaluation diagnostics: `.../diagnostics/`
- prediction exports: `.../model_output/`
- plots: `.../plots/`

Reusing the same `run.name` overwrites that run's files. Use a new name to preserve earlier results.

`evaluation/plot_evaluation.py` reads from the run folder matching `run.name` in the config and writes plots back to that run's `plots/` folder.

## Reproducibility

Each run writes a `run_metadata.yaml` in the run root and updates `output/08_baseline_results/latest_run.txt`. Evaluation CSVs carry a `sample_pct` column so subsampled runs remain comparable.

## Notes

**GPU usage monitoring:**
```bash
nvidia-smi -L
watch -n 1 nvidia-smi
```
or
```bash
nvitop
```
after installing with `pip install nvitop`.

**CPU usage monitoring (Python processes):**
```bash
htop -p $(pgrep -d',' -f python)
```

**Run without crash on logout** (add `-u` for unbuffered stdout so output appears in the log file):
```bash
nohup python -u evaluation/tune_hyperparams.py > tune_log.txt 2>&1 &
```
```bash
nohup python -u evaluation/run_evaluation.py > tune_log.txt 2>&1 &
```



read `tune_log.txt` for output and errors. Use `tail -f tune_log.txt` to monitor live.
Monitor live output with `tail -f tune_log.txt`.
