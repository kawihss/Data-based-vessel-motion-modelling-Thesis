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

## Model Layer

### `models/base_model.py`

`BaselineModel` is the common interface. Evaluator logic is model-agnostic and only depends on `predict(context_df, n_pred_steps)`.

### `models/kinematic.py`

Current baseline models:

- `ConstantVelocityModel`
- `ConstantTurnRateVelocityModel` (CTRV)

Both models use one parameterization mode:

1. `velocity_steps` (integer number of recent context steps to average).

## Evaluation Layer

### `evaluation/metrics.py`

Provides trajectory quality metrics:

- ADE
- FDE
- RMSE
- ADE per prediction step

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

## Hyperparameter Tuning (Optuna)

### `evaluation/tune_hyperparams.py`

Purpose:

- tune `velocity_steps` on the **validation split** for CV and CTRV.

Current behavior:

1. loads cached numpy tracks from `output/07_parquet/` (validation split),
2. runs Optuna TPE optimization per model,
3. prints per-trial RMSE,
4. writes trial CSVs and summary CSV,
5. writes RMSE-vs-`velocity_steps` plot per model.

Outputs:

- `evaluation/diagnostics/tuning_cv_val.csv`
- `evaluation/diagnostics/tuning_ctrv_val.csv`
- `evaluation/diagnostics/tuning_best_params_val.csv`
- `evaluation/diagnostics/tuning_all_trials_val.csv`
- `evaluation/diagnostics/tuning_cv_val_plot.png`
- `evaluation/diagnostics/tuning_ctrv_val_plot.png`

Important split policy:

- validation is for parameter selection,
- test is only for final reporting.

## Final Baseline Evaluation (Test)

### `evaluation/run_evaluation.py`

Purpose:

- run final metrics on test split for selected models,
- export diagnostics and optional prediction trajectories.

Current behavior:

1. tries to load tuned parameters from `tuning_best_params_val.csv`,
2. falls back to defaults if tuning summary is missing,
3. evaluates on configured split (default: `test`),
4. writes outputs under `output/08_baseline_results/`.

Current baseline output structure:

- diagnostics: `output/08_baseline_results/diagnostics/`
- prediction exports: `output/08_baseline_results/model_output/`

## Recommended Usage Sequence

1. run `evaluation/tune_hyperparams.py` on validation,
2. verify `tuning_best_params_val.csv`,
3. run `evaluation/run_evaluation.py` on test,
4. compare final model metrics from the model comparison CSV.