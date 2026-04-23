import numpy as np
import pandas as pd
import re
import csv
import time
from pathlib import Path
from .metrics import evaluate_trajectory, plot_ade_over_horizon
from models.kinematic import ConstantVelocityModel

# Valid context labels from the preprocessing pipeline
CONTEXT_LABELS = {'river', 'channel', 'harbour', 'lock', 'unknown'}
MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})_\d{2}$")

TRACK_USECOLS = ['track_id', 'role', 't_utc', 'x', 'y', 'rot', 'vessel_id', 'context']
TRACK_DTYPES = {
    'track_id': 'int64',
    'role': 'category',
    'x': 'float32',
    'y': 'float32',
    'rot': 'float32',
    'vessel_id': 'string',
    'context': 'category',
}


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


def _extract_month_label(file_path):
    name = Path(file_path).stem
    match = MONTH_PATTERN.search(name)
    if match is not None:
        year, month = match.groups()
        return f"{year}-{month}"
    return None


def _read_track_file(file_path):
    return pd.read_csv(
        file_path,
        usecols=TRACK_USECOLS,
        dtype=TRACK_DTYPES,
        low_memory=False,
    )


def _iter_track_groups(df, with_track_id=False):
    for track_id, group in df.groupby('track_id', sort=False):
        context = group[group['role'] == 'context']
        pred = group[group['role'] == 'prediction']
        if len(context) == 0 or len(pred) == 0:
            continue
        if with_track_id:
            yield track_id, context, pred
        else:
            yield context, pred


def load_tracks(data_dir, split='test', context_filter=None):
    # Generator that yields (context_df, pred_df) one track at a time.
    # Reads one file at a time so only one CSV is in memory at once.
    # split: 'train', 'val', or 'test'
    # context_filter: str or list of str from CONTEXT_LABELS, e.g. 'lock' or ['harbour', 'lock']
    files = _resolve_files(data_dir, split, context_filter)
    print(f"Streaming {len(files)} file(s) for split '{split}'" +
          (f", context(s) {context_filter}" if context_filter else ""))

    for f in files:
        df = _read_track_file(f)
        yield from _iter_track_groups(df)


def load_tracks_cached(data_dir, split='test', context_filter=None):
    # Materialize tracks in memory once for repeated evaluations (e.g., hyperparameter tuning).
    files = _resolve_files(data_dir, split, context_filter)
    print(f"Caching tracks from {len(files)} file(s) for split '{split}'" +
          (f", context(s) {context_filter}" if context_filter else ""))

    cached_tracks = []
    for f in files:
        df = _read_track_file(f)
        for context, pred in _iter_track_groups(df):
            # Copy slices once so repeated trials don't keep dataframe views alive.
            cached_tracks.append((context.copy(), pred.copy()))

    print(f"Cached {len(cached_tracks)} track(s) in memory")
    return cached_tracks


def load_tracks_cached_numpy(data_dir, split='test', context_filter=None):
    files = _resolve_files(data_dir, split, context_filter)
    print(f"Caching numpy tracks from {len(files)} file(s) for split '{split}'" +
          (f", context(s) {context_filter}" if context_filter else ""))

    cached_tracks = []
    for f in files:
        df = _read_track_file(f)
        for context_df, pred_df in _iter_track_groups(df):
            ctx = context_df.sort_values('t_utc')
            pred = pred_df.sort_values('t_utc')

            x_ctx = ctx['x'].to_numpy(dtype=float, copy=True)
            y_ctx = ctx['y'].to_numpy(dtype=float, copy=True)
            rot_ctx = ctx['rot'].to_numpy(dtype=float, copy=True)
            true_xy = pred[['x', 'y']].to_numpy(dtype=float, copy=True)

            cached_tracks.append({
                'x_ctx': x_ctx,
                'y_ctx': y_ctx,
                'rot_ctx': rot_ctx,
                'last_x': float(x_ctx[-1]),
                'last_y': float(y_ctx[-1]),
                'true_xy': true_xy,
                'n_pred_steps': int(len(true_xy)),
            })

    print(f"Cached {len(cached_tracks)} numpy track(s) in memory")
    return cached_tracks


def load_tracks_from_file(file_path):
    df = _read_track_file(file_path)
    yield from _iter_track_groups(df)


def reconstruct_positions(last_x, last_y, displacements):
    # displacements: (n_steps, 2) array of predicted (dx, dy)
    # returns absolute (x, y) positions: (n_steps, 2)
    positions = np.cumsum(displacements, axis=0)
    positions[:, 0] += last_x
    positions[:, 1] += last_y
    return positions


def _default_model_key(model):
    name = getattr(model, 'name', 'model')
    key = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
    return key or 'model'


def export_predictions_for_file(model, file_path, output_dir, split='test', model_key=None, model_label=None):
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_key = model_key or _default_model_key(model)
    resolved_model_label = model_label or getattr(model, 'name', resolved_model_key)
    output_path = output_dir / f"{file_path.stem}__{resolved_model_key}.csv"

    columns = [
        'source_file',
        'source_stem',
        'split',
        'model_key',
        'model_label',
        'track_id',
        'vessel_id',
        'context',
        'pred_step',
        't_utc_pred',
        'x_pred',
        'y_pred',
        'x_gt',
        'y_gt',
    ]

    row_count = 0
    df = _read_track_file(file_path)

    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for track_id, context_df, pred_df in _iter_track_groups(df, with_track_id=True):

            if context_df.empty or pred_df.empty:
                continue

            n_pred_steps = len(pred_df)
            last_x = context_df['x'].iloc[-1]
            last_y = context_df['y'].iloc[-1]

            displacements = model.predict(context_df, n_pred_steps)
            pred_positions = reconstruct_positions(last_x, last_y, displacements)

            if len(pred_positions) != n_pred_steps:
                continue

            vessel_id = context_df['vessel_id'].iloc[-1] if 'vessel_id' in context_df else None
            context_label = context_df['context'].iloc[-1] if 'context' in context_df else None

            pred_df = pred_df.sort_values('t_utc').reset_index(drop=True)

            for i in range(n_pred_steps):
                writer.writerow({
                    'source_file': file_path.name,
                    'source_stem': file_path.stem,
                    'split': split,
                    'model_key': resolved_model_key,
                    'model_label': resolved_model_label,
                    'track_id': track_id,
                    'vessel_id': vessel_id,
                    'context': context_label,
                    'pred_step': i + 1,
                    't_utc_pred': pred_df['t_utc'].iloc[i] if 't_utc' in pred_df else '',
                    'x_pred': float(pred_positions[i, 0]),
                    'y_pred': float(pred_positions[i, 1]),
                    'x_gt': float(pred_df['x'].iloc[i]),
                    'y_gt': float(pred_df['y'].iloc[i]),
                })
                row_count += 1

    return {'output_path': output_path, 'rows': row_count}


def _evaluate_tracks(model, tracks):
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

    if not all_true:
        empty_true = np.empty((0, 0, 2))
        empty_metrics = {
            'ADE': np.nan,
            'FDE': np.nan,
            'RMSE': np.nan,
            'ADE_per_step': np.array([]),
            'n_tracks': 0,
        }
        return empty_metrics, empty_true, empty_true.copy()

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    metrics = evaluate_trajectory(all_true, all_pred)
    metrics['n_tracks'] = len(all_true)
    return metrics, all_true, all_pred


def evaluate_model(model, tracks):
    # Core evaluation loop, Optuna objective calls this directly.
    # tracks: generator or list of (context_df, pred_df) from load_tracks()
    # Accumulates only small numpy arrays, not full DataFrames, to stay memory efficient.
    # returns dict with ADE, FDE, RMSE, n_tracks
    metrics, _, _ = _evaluate_tracks(model, tracks)
    return metrics


def evaluate_model_cached(model, cached_tracks):
    # Evaluate using a preloaded list of tracks to avoid repeated file I/O in tuning loops.
    metrics, _, _ = _evaluate_tracks(model, cached_tracks)
    return metrics


def evaluate_model_cached_numpy(model, cached_tracks):
    all_true = []
    all_pred = []

    for track in cached_tracks:
        n_pred_steps = track['n_pred_steps']
        if n_pred_steps <= 0:
            continue

        if hasattr(model, 'predict_from_arrays'):
            displacements = model.predict_from_arrays(
                track['x_ctx'],
                track['y_ctx'],
                track['rot_ctx'],
                n_pred_steps,
            )
        else:
            # Fallback keeps compatibility with older models.
            context_df = pd.DataFrame({
                'x': track['x_ctx'],
                'y': track['y_ctx'],
                'rot': track['rot_ctx'],
            })
            displacements = model.predict(context_df, n_pred_steps)

        pred_positions = reconstruct_positions(track['last_x'], track['last_y'], displacements)
        true_positions = track['true_xy']

        if len(pred_positions) != len(true_positions):
            continue

        all_true.append(true_positions)
        all_pred.append(pred_positions)

    if not all_true:
        return {
            'ADE': np.nan,
            'FDE': np.nan,
            'RMSE': np.nan,
            'ADE_per_step': np.array([]),
            'n_tracks': 0,
        }

    metrics = evaluate_trajectory(np.array(all_true), np.array(all_pred))
    metrics['n_tracks'] = len(all_true)
    return metrics


def benchmark_loading(data_dir, split='val', context_filter=None, repeats=3):
    rows = []
    for i in range(repeats):
        t0 = time.perf_counter()
        tracks_df = load_tracks_cached(data_dir, split=split, context_filter=context_filter)
        t1 = time.perf_counter()
        tracks_np = load_tracks_cached_numpy(data_dir, split=split, context_filter=context_filter)
        t2 = time.perf_counter()
        rows.append({
            'run': i + 1,
            'tracks_df': len(tracks_df),
            'tracks_np': len(tracks_np),
            'load_df_s': t1 - t0,
            'load_np_s': t2 - t1,
            'total_s': t2 - t0,
        })
    return pd.DataFrame(rows)


def evaluate_file_metrics(model, data_dir, split='test', context_filter=None):
    files = _resolve_files(data_dir, split, context_filter)
    rows = []

    print(f"Evaluating {len(files)} file(s) individually for split '{split}'")

    for file_path in files:
        metrics, _, _ = _evaluate_tracks(model, load_tracks_from_file(file_path))
        row = {
            'file': Path(file_path).name,
            'month': _extract_month_label(file_path),
            'ADE': metrics['ADE'],
            'FDE': metrics['FDE'],
            'RMSE': metrics['RMSE'],
            'n_tracks': metrics['n_tracks'],
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(['RMSE', 'FDE', 'ADE'], ascending=False).reset_index(drop=True)


def plot_horizon_error(metrics, label=None, ax=None, step_duration_s=30, save_path=None):
    # wrapper: pass the dict returned by evaluate_model/run_evaluation
    return plot_ade_over_horizon(
        metrics['ADE_per_step'],
        step_duration_s=step_duration_s,
        label=label,
        ax=ax,
        save_path=save_path,
    )


def run_evaluation(
    model,
    data_dir,
    split='test',
    context_filter=None,
    export_predictions=False,
    prediction_output_dir=None,
    model_key=None,
    model_label=None,
):
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
    files = _resolve_files(data_dir, split, context_filter)
    print(f"Streaming {len(files)} file(s) for split '{split}'" +
          (f", context(s) {context_filter}" if context_filter else ""))

    prediction_exports = []
    if export_predictions:
        if prediction_output_dir is None:
            prediction_output_dir = Path(data_dir).resolve().parent / '07_model_output'
        prediction_output_dir = Path(prediction_output_dir)
        prediction_output_dir.mkdir(parents=True, exist_ok=True)

    per_file_rows = []
    all_true = []
    all_pred = []

    for file_path in files:
        df = _read_track_file(file_path)
        grouped_tracks = list(_iter_track_groups(df, with_track_id=True))
        file_metrics, file_true, file_pred = _evaluate_tracks(
            model,
            ((context_df, pred_df) for _, context_df, pred_df in grouped_tracks),
        )
        per_file_rows.append({
            'file': Path(file_path).name,
            'month': _extract_month_label(file_path),
            'ADE': file_metrics['ADE'],
            'FDE': file_metrics['FDE'],
            'RMSE': file_metrics['RMSE'],
            'n_tracks': file_metrics['n_tracks'],
        })

        if file_metrics['n_tracks'] > 0:
            all_true.extend(file_true)
            all_pred.extend(file_pred)

        if export_predictions:
            output_path = prediction_output_dir / f"{Path(file_path).stem}__{(model_key or _default_model_key(model))}.csv"
            columns = [
                'source_file',
                'source_stem',
                'split',
                'model_key',
                'model_label',
                'track_id',
                'vessel_id',
                'context',
                'pred_step',
                't_utc_pred',
                'x_pred',
                'y_pred',
                'x_gt',
                'y_gt',
            ]
            row_count = 0
            with output_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                for track_id, context_df, pred_df in grouped_tracks:
                    n_pred_steps = len(pred_df)
                    last_x = context_df['x'].iloc[-1]
                    last_y = context_df['y'].iloc[-1]

                    displacements = model.predict(context_df, n_pred_steps)
                    pred_positions = reconstruct_positions(last_x, last_y, displacements)
                    if len(pred_positions) != n_pred_steps:
                        continue

                    vessel_id = context_df['vessel_id'].iloc[-1] if 'vessel_id' in context_df else None
                    context_label = context_df['context'].iloc[-1] if 'context' in context_df else None
                    pred_sorted = pred_df.sort_values('t_utc').reset_index(drop=True)

                    for i in range(n_pred_steps):
                        writer.writerow({
                            'source_file': Path(file_path).name,
                            'source_stem': Path(file_path).stem,
                            'split': split,
                            'model_key': model_key or _default_model_key(model),
                            'model_label': model_label or getattr(model, 'name', _default_model_key(model)),
                            'track_id': track_id,
                            'vessel_id': vessel_id,
                            'context': context_label,
                            'pred_step': i + 1,
                            't_utc_pred': pred_sorted['t_utc'].iloc[i] if 't_utc' in pred_sorted else '',
                            'x_pred': float(pred_positions[i, 0]),
                            'y_pred': float(pred_positions[i, 1]),
                            'x_gt': float(pred_sorted['x'].iloc[i]),
                            'y_gt': float(pred_sorted['y'].iloc[i]),
                        })
                        row_count += 1

            prediction_exports.append({
                'file': Path(file_path).name,
                'output_path': str(output_path),
                'rows': row_count,
            })

    if all_true:
        metrics = evaluate_trajectory(np.array(all_true), np.array(all_pred))
        metrics['n_tracks'] = len(all_true)
    else:
        metrics = {
            'ADE': np.nan,
            'FDE': np.nan,
            'RMSE': np.nan,
            'ADE_per_step': np.array([]),
            'n_tracks': 0,
        }

    per_file_metrics = pd.DataFrame(per_file_rows)
    if not per_file_metrics.empty:
        per_file_metrics = per_file_metrics.sort_values(['RMSE', 'FDE', 'ADE'], ascending=False).reset_index(drop=True)

    metrics['per_file_metrics'] = per_file_metrics

    if not per_file_metrics.empty and per_file_metrics['month'].notna().any():
        per_month_metrics = (
            per_file_metrics
            .groupby('month', as_index=False)
            .agg(
                ADE=('ADE', 'mean'),
                FDE=('FDE', 'mean'),
                RMSE=('RMSE', 'mean'),
                n_tracks=('n_tracks', 'sum'),
                n_files=('file', 'count'),
            )
            .sort_values('RMSE', ascending=False)
            .reset_index(drop=True)
        )
    else:
        per_month_metrics = pd.DataFrame(columns=['month', 'ADE', 'FDE', 'RMSE', 'n_tracks', 'n_files'])

    metrics['per_month_metrics'] = per_month_metrics

    if export_predictions:
        metrics['prediction_exports'] = pd.DataFrame(prediction_exports)

    return metrics


