# Evaluation Framework and Current Workflow

This document describes the current evaluation and hyperparameter tuning state for the baseline motion models.

## Position in the Pipeline

After preprocessing, evaluation consumes windowed trajectory data and produces:

1. aggregate metrics (ADE, FDE, RMSE),
2. per-file and per-month diagnostics,
3. optional prediction exports for visual inspection.

Current runtime data paths:

- tuning input: `output/07_parquet/`
- final baseline evaluation input: `output/07_parquet/`
- baseline outputs: `output/08_baseline_results/`

## System requirements

- Python dependencies: see `requirements.txt`
- NVIDIA CUDA 8.0 or later for the tirex library
- RAM: 32+ GB recommended (tested on 48GB+ machines) for caching track data during tuning and evaluation
- Chronos-2 on Linux currently requires `export HF_HUB_DISABLE_XET=1` before running evaluation or tuning so model downloads work reliably
- **TiRex CUDA backend:** `xlstm` calls `torch.utils.cpp_extension.include_paths(cuda=True)` which was removed in PyTorch 2.11+. Patch cuda_init.py` replace `include_paths(cuda=True)` with `include_paths()` and hardcode `CUDA_LIB` to your CUDA lib path (e.g. `/usr/local/cuda-12.6/targets/x86_64-linux/lib`). This patch must be reapplied after `pip install --upgrade xlstm`. As a fallback if CUDA kernel compilation fails, set `backend: torch` in `configs/evaluation.yaml` under `models.tirex` to uses PyTorch ops on GPU without custom kernels (3-4 times slower).


## Model Layer

### `models/base_model.py`

`BaselineModel` is the common interface. Evaluator logic is model-agnostic and only depends on `predict(context_df, n_pred_steps)`. Sequence models are

### `models/kinematic.py`

Current baseline models:

- `ConstantVelocityModel` (CV)
- `ConstantTurnRateVelocityModel` (CTRV)
- `HybridCVCTRVModel` (Hybrid)
- `KalmanFilter` (Kalman with CV state model)
- `Chronos2ZeroShotModel` (Chronos-2 base, zero-shot, fixed 10-step context and 10-step prediction horizon, no tuning)

CV and CTRV use one parameterization:

- `velocity_steps`: number of recent context steps to average.

The Hybrid model selects between CV and CTRV per track based on a rotation-rate threshold. It has additional parameters `cv_velocity_steps`, `ctrv_velocity_steps`, `rot_steps`, and `rot_threshold`.

The Kalman model uses a linear constant-velocity state transition and is evaluated in two phases:

- context phase: regular Kalman prediction-correction cycle with measurements,
- prediction phase: pure model rollout without further measurement correction.

Kalman tuning parameters are `q_pos`, `q_vel`, `r_pos`, `p0_pos`, `p0_vel`, and `init_velocity_steps`.

**Note on Hybrid model behavior:** In practice, the Hybrid model almost always selects the CV branch, even after tuning. The rotation threshold ends up rarely triggered on this dataset, so Hybrid predictions differ from pure CV only in marginal cases. Whether to include the Hybrid model in the final thesis evaluation is still open; it adds complexity for negligible empirical gain.

## Evaluation Layer

### `evaluation/metrics.py`

Provides trajectory quality metrics:

- ADE
- FDE
- RMSE
- ADE per prediction step
- MIW from the `[q=0.1, q=0.9]` interval on predicted `dx`/`dy` displacements
- Coverage of the `[q=0.1, q=0.9]` interval on predicted `dx`/`dy` displacements
- Additional quantile-derived metrics for Chronos-2: pinball loss, CRPS approximation, Winkler interval score

### `evaluation/evaluator.py`

Core responsibilities:

- resolve files by split/context,
- load and group tracks,
- call model prediction,
- reconstruct positions,
- aggregate metrics,
- return diagnostic tables.

Two execution modes are currently available:

1. streaming mode (file-by-file),
2. cached numpy mode for fast repeated evaluation during tuning.

Cached mode is used by Optuna tuning to avoid repeated disk I/O per trial.

## Hyperparameter Tuning

### `evaluation/tune_hyperparams.py`

Purpose:

- tune hyperparameters on the **validation split** for all enabled models.

**1D models — full grid search (exhaustive):**

CV, CTRV, CTRV Arc, and TiRex LSTM each have one tuned parameter (`velocity_steps`). The tuner iterates over every integer from 1 to `max_velocity_steps` (derived from the longest validation context window) and picks the step count with the lowest RMSE. Each model runs in a separate process.

**Multi-parameter models — Optuna TPE:**

- Hybrid CV/CTRV: 4 parameters (`cv_velocity_steps`, `ctrv_velocity_steps`, `rot_steps`, `rot_threshold`), seeded with the grid-search results for CV and CTRV.
- Kalman: 5 parameters (`q_pos`, `q_vel`, `r_pos`, `p0_pos`, `p0_vel`).
- CTRV EKF: 8 parameters (noise/covariance values + `init_velocity_steps`), with early stopping.

Current behavior:

1. starts 1D model grid-search jobs in parallel processes,
2. caches validation tracks in RAM inside each worker process,
3. collects results and writes intermediate `tuning_best_params_val.csv`,
4. runs Hybrid, Kalman, and CTRV EKF tuning sequentially (Hybrid seeds from CV/CTRV grid results),
5. writes final combined best-params and all-trials CSVs.

Outputs:

- `tuning_cv_val.csv`, `tuning_ctrv_val.csv`, `tuning_ctrv_arc_val.csv`, `tuning_tirex_lstm_val.csv`
- `tuning_hybrid_cv_ctrv_val.csv`, `tuning_kalman_val.csv`, `tuning_ctrv_ekf_val.csv`
- `tuning_best_params_val.csv`
- `tuning_all_trials_val.csv`

All outputs are written under the run folder: `output/08_baseline_results/runs/{run.name}/tuning/`.

Tuning plots are generated by `evaluation/plot_evaluation.py` from the saved CSV files.

Important split policy:

- validation is for parameter selection,
- test is only for final reporting.

## Final Baseline Evaluation (Test)

### `evaluation/run_evaluation.py`

Purpose:

- run final metrics on test split for selected models,
- export diagnostics and optional prediction trajectories.

Current behavior:

1. loads tuned parameters from `tuning_best_params_val.csv` (crashes if missing; run tuning first),
2. loads all selected parquet files into memory once and reuses them across all enabled models,
3. evaluates on configured split (default: `test`),
4. computes and writes per-file and per-month metrics,
5. optionally exports per-model prediction CSVs,
6. writes outputs under `output/08_baseline_results/`.

Chronos-2 differs from the tuned baseline models in one important way:

1. it runs strictly zero-shot,
2. it does not read hyperparameters from `tuning_best_params_val.csv`,
3. it uses `q=0.5` as the point forecast,
4. it writes per-context diagnostics for `harbour`, `river`, `channel`, and `lock`,
5. it exports quantile-aware diagnostics and channel-importance tables when enabled.

On Linux, set `HF_HUB_DISABLE_XET=1` in the shell before running Chronos-2, for example with `export HF_HUB_DISABLE_XET=1`.

Current baseline output structure:

- diagnostics: `output/08_baseline_results/diagnostics/`
- prediction exports: `output/08_baseline_results/model_output/`

## Plotting and Reporting

### `evaluation/plot_evaluation.py`

Purpose:

- generate publication/reporting plots from existing CSV diagnostics,
- avoid rerunning model inference just to regenerate figures.

Current behavior:

1. reads test diagnostics from `output/08_baseline_results/diagnostics/`,
2. creates ADE-over-horizon plot per enabled model,
3. creates monthly ADE/FDE/RMSE plots per enabled model,
4. reads tuning summaries from `evaluation/diagnostics/` and regenerates CV/CTRV/Hybrid tuning plots,
5. writes all plot images to `output/08_baseline_results/plots/`.

Model visibility for plots is controlled by the top-level flags in the script (`PLOT_CONSTANT_VELOCITY`, `PLOT_CTRV`, `PLOT_HYBRID`).

## Recommended Usage Sequence

1. run `evaluation/tune_hyperparams.py` on validation,
2. verify `tuning_best_params_val.csv`,
3. run `evaluation/run_evaluation.py` on test,
4. run `evaluation/plot_evaluation.py` to generate evaluation and tuning plots,
5. compare final model metrics and generated figures.

## Central Config (YAML)

Runtime settings are configured in `configs/evaluation.yaml`.

This single file controls:

- run metadata (`run.name`, `run.description`, `run.seed`, `run.sample_pct`),
- sampling (`sampling.tuning_validation_pct`, `sampling.evaluation_pct`),
- data settings (`data.parquet_dir`, `data.split`, `data.context_filter`),
- model enable switches (`models.*`),
- tuning hyperparameters (`tuning.*`),
- plotting model visibility and step duration (`plotting.*`).

`sampling.tuning_validation_pct` controls file-level subsampling of the validation split during hyperparameter optimization.

`sampling.evaluation_pct` controls file-level subsampling of the configured evaluation split (usually test), useful for faster debug/test evaluation runs.

All sampling is deterministic and reproducible via `run.seed`.

## Non-overwriting Output Layout

Outputs are now versioned per run name to avoid data loss:

- base: `output/08_baseline_results/runs/{run.name}/`
- tuning CSVs: `.../tuning/`
- evaluation diagnostics: `.../diagnostics/`
- prediction exports: `.../model_output/`
- plots: `.../plots/`

If you reuse the same `run.name`, tuning and evaluation overwrite files in that run folder.
Use a new `run.name` when you want to preserve earlier results.

## Plotting Source

`evaluation/plot_evaluation.py` now reads from the newest run folder under `output/08_baseline_results/runs/` by modification time and writes plots back to that run's `plots/` folder.

## Reproducibility Metadata

Each run writes:

- `run_metadata.yaml` in the run root (contains run name, seed, sample percentage, split/context),
- `output/08_baseline_results/latest_run.txt` updated to the current run name.

Evaluation CSVs also carry `sample_pct` so future subsampling runs can be compared without changing schema.



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
after installing with `pip install nvitop`

**CPU usage monitoring (Python processes):**
```bash
htop -p $(pgrep -d',' -f python)
```


**Run without crash on logout:**

add tag -u for unbuffered stoudt if you want to see live output in the log file:
```bash
nohup python -u evaluation/tune_hyperparams.py > tune_log.txt 2>&1 &
```
```bash
nohup python -u evaluation/run_evaluation.py > tune_log.txt 2>&1 &
```



read `tune_log.txt` for output and errors. Use `tail -f tune_log.txt` to monitor live.
