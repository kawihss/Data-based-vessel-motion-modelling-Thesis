import numpy as np
import pandas as pd
from pathlib import Path
from .metrics import evaluate_trajectory, plot_ade_over_horizon
from models.kinematic import ConstantVelocityModel

# Valid context labels from the preprocessing pipeline
CONTEXT_LABELS = {'river', 'channel', 'harbour', 'lock', 'unknown'}


def _resolve_files(data_dir, split, context_filter):
    data_dir = Path(data_dir)
    if context_filter is not None:
        if isinstance(context_filter, str):
            context_filter = [context_filter]
        patterns = [f"{split}_{ctx}_*.csv" for ctx in context_filter]
    else:
        patterns = [f"{split}_*.csv"]

    files = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(f"No files found for split '{split}' in {data_dir}")
    return files


def load_tracks(data_dir, split='test', context_filter=None):
    # Generator that yields (context_df, pred_df) one track at a time.
    # Reads one file at a time so only one CSV is in memory at once.
    # split: 'train', 'val', or 'test'
    # context_filter: str or list of str from CONTEXT_LABELS, e.g. 'lock' or ['harbour', 'lock']
    files = _resolve_files(data_dir, split, context_filter)
    print(f"Streaming {len(files)} file(s) for split '{split}'" +
          (f", context(s) {context_filter}" if context_filter else ""))

    for f in files:
        df = pd.read_csv(f, low_memory=False)
        for _, group in df.groupby('track_id', sort=False):
            context = group[group['role'] == 'context']
            pred = group[group['role'] == 'prediction']
            if len(context) > 0 and len(pred) > 0:
                yield context, pred


def reconstruct_positions(last_x, last_y, displacements):
    # displacements: (n_steps, 2) array of predicted (dx, dy)
    # returns absolute (x, y) positions: (n_steps, 2)
    positions = np.cumsum(displacements, axis=0)
    positions[:, 0] += last_x
    positions[:, 1] += last_y
    return positions


def evaluate_model(model, tracks):
    # Core evaluation loop, Optuna objective calls this directly.
    # tracks: generator or list of (context_df, pred_df) from load_tracks()
    # Accumulates only small numpy arrays, not full DataFrames, to stay memory efficient.
    # returns dict with ADE, FDE, RMSE, n_tracks
    all_true = []
    all_pred = []

    for context_df, pred_df in tracks:
        n_pred_steps = len(pred_df)
        last_x = context_df['x'].iloc[-1]
        last_y = context_df['y'].iloc[-1]

        displacements = model.predict(context_df, n_pred_steps)
        pred_positions = reconstruct_positions(last_x, last_y, displacements)
        true_positions = pred_df[['x', 'y']].values

        if len(true_positions) != n_pred_steps:
            continue

        all_true.append(true_positions)
        all_pred.append(pred_positions)

    all_true = np.array(all_true)  # (n_tracks, n_steps, 2)
    all_pred = np.array(all_pred)

    metrics = evaluate_trajectory(all_true, all_pred)
    metrics['n_tracks'] = len(all_true)
    return metrics


def plot_horizon_error(metrics, label=None, ax=None, step_duration_s=30, save_path=None):
    # wrapper: pass the dict returned by evaluate_model/run_evaluation
    return plot_ade_over_horizon(
        metrics['ADE_per_step'],
        step_duration_s=step_duration_s,
        label=label,
        ax=ax,
        save_path=save_path,
    )


def run_evaluation(model, data_dir, split='test', context_filter=None):
    # wrapper: load all files for a split and evaluate.
    # data_dir: path to output/05_normalized/
    # split: 'train', 'val', or 'test' (kinematic models always use 'test')
    # context_filter: str or list of str, e.g. 'lock', 'harbour', ['river', 'channel']
    #   None = evaluate on all contexts
    #
    # Optuna objective example (lock-specific):
    #   def objective(trial):
    #       frac = trial.suggest_float('velocity_fraction', 0.1, 1.0)
    #       model = ConstantVelocityModel(velocity_fraction=frac)
    #       return run_evaluation(model, 'output/05_normalized', context_filter='lock')['ADE']
    tracks = load_tracks(data_dir, split=split, context_filter=context_filter)
    return evaluate_model(model, tracks)


