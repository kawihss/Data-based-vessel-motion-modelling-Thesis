# Evaluation Framework and Model Architecture

>This file is not user-ready and only documents the current state at a high level.


## Position in the Pipeline

The preprocessing pipeline ends with normalized context/prediction windows in `output/05_normalized/`. From that point on, the repository moves into the model and evaluation stage.

Conceptually, this stage is split into two parts:

1. `models/` contains prediction models and the places where additional model families will be added.
2. `evaluation/` contains data loading, prediction execution, metric computation, and diagnostic output.

The idea is simple: preprocessing standardizes the data, and the evaluation framework measures how well a model predicts on those standardized samples.

## Object-Oriented Approach

The current OOP structure is intentionally simple:

- `models/base_model.py` defines the shared abstraction through `BaselineModel`.
- Concrete models inherit from that base class and implement `predict(...)`.
- The evaluator does not depend on model internals. It only expects a compatible model object.

This keeps evaluation tied to a stable interface rather than to one specific implementation. That is the key requirement if the project later grows from simple baselines to filter-based or learned sequence models.

## Current State of `models/`

At the moment, the model layer is more of a scaffold than a finished framework.

### `models/base_model.py`

`BaselineModel` is the common parent class. Right now it mainly stores the model name and defines the expectation that child classes implement `predict(...)`.

This is the main extension point for everything that will be added later.

### `models/kinematic.py`

This file currently contains the only concrete model: `ConstantVelocityModel`.


- sort the context track by time,
- compute recent displacements from `x` and `y`,
- average the last fraction of those steps,
- repeat that mean displacement across the prediction horizon.


### `models/filters.py`

 Kalman Filter / Extended Kalman Filter variants with CV or CTRV dynamics may later be implemented.


### `models/sequence/`



## Current State of `evaluation/`

The evaluation package is already more concrete than the model package. This is where the reusable execution and metric logic currently lives.

### `evaluation/metrics.py`

This module contains the core trajectory metrics:

- `ADE`: average displacement error across all tracks and all prediction steps,
- `FDE`: displacement error at the final prediction step,
- `RMSE`: root mean squared error over reconstructed positions,
- `ADE_per_step`: error development over the prediction horizon.

It also provides a helper for plotting the ADE curve over time. 

### `evaluation/evaluator.py`

This is the central orchestration module.

Its current responsibilities are:

- locating normalized data files by split and optional context filter,
- streaming tracks from CSV files instead of loading the full dataset at once,
- separating grouped tracks into context and prediction parts,
- calling `model.predict(context_df, n_pred_steps)`,
- reconstructing absolute future positions from predicted displacements,
- aggregating metrics across tracks, files, and months.

An important detail is that the evaluator is already designed to be relatively memory-efficient. It processes one file and one track at a time and only keeps the smaller arrays needed for metric computation.

The module exposes several entry points with slightly different purposes:

- `evaluate_model(...)` for aggregate evaluation only,
- `evaluate_file_metrics(...)` for per-file reporting,
- `run_evaluation(...)` as the broader wrapper including per-file and per-month diagnostics,
- `plot_horizon_error(...)` as a convenience wrapper for the horizon plot.

This is the main layer between the processed data and the models.

### `evaluation/run_evaluation.py`

This file currently serves as the executable example of the intended workflow.

The current flow is:

1. instantiate a model,
2. point the evaluator to `output/05_normalized`,
3. run evaluation on a split,
4. inspect or export metrics and diagnostics.

In its current form, this script documents the practical workflow better than any external documentation.

### `evaluation/diagnostics/`

This directory is intended to store generated evaluation artifacts such as per-file and per-month metric tables.

## Data Flow After Preprocessing

The intended flow after preprocessing is currently:

1. preprocessing writes normalized segments to `output/05_normalized/`,
2. a model consumes the context portion of a track,
3. the model returns predicted displacements for the prediction horizon,
4. the evaluator reconstructs absolute target positions,
5. the metric layer compares prediction and ground truth,
6. diagnostics are written as tables or plots.

This keeps responsibilities clearly separated:

- preprocessing prepares the data,
- models generate forecasts,
- evaluation measures prediction quality.

## Next Steps

The immediate next implementation steps are:

1. implement a CTRV model as the next kinematic baseline,
2. implement the two planned filters in `models/filters.py`,
3. add predictions to visualizer
