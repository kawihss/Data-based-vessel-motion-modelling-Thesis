import numpy as np
import pandas as pd
import re
import csv
from pathlib import Path
from .metrics import evaluate_trajectory, plot_ade_over_horizon
from .runtime_config import subsample_items

# Valid context labels from the preprocessing pipeline
CONTEXT_LABELS = {'river', 'channel', 'harbour', 'lock', 'unknown'}
MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})_\d{2}$")

TRACK_USECOLS = ['track_id', 'role', 't_utc', 'x', 'y', 'rot', 'vessel_id', 'context', 'dx_norm', 'dy_norm', 'sog_norm', 'cog_sin_norm', 'cog_cos_norm', 'dt_norm', 'rot_norm']

#internal functions marked _name

def _resolve_files(data_dir, split, context_filter):
    #looks for files in the data_dir, filtered by split and context if provided, 
    # returns list of file paths
    data_dir = Path(data_dir)
    if context_filter is not None:
        if isinstance(context_filter, str):
            context_filter = [context_filter]
        base_patterns = [f"{split}_{ctx}_*" for ctx in context_filter]
    else:
        base_patterns = [f"{split}_*"]

    selected_files = []
    for base_pattern in base_patterns:
        selected_files.extend(sorted(data_dir.glob(f"{base_pattern}.parquet")))

    if not selected_files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' in {data_dir}. "
            "Run step 07_generate_parquet.py first."
        )
    return selected_files


def _extract_month_label(file_path):
    name = Path(file_path).stem
    match = MONTH_PATTERN.search(name)    #regex 
    year, month = match.groups()
    return f"{year}-{month}"


def _read_track_file(file_path):
    #reads required columns from a track file, returns dataframe
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.parquet':
        return pd.read_parquet(file_path, columns=TRACK_USECOLS)

    raise ValueError(f"Unsupported file type for track loading: {file_path}")


def _iter_track_groups(df, with_track_id=False):
    #iterates over tracks dataframe, yields context and prediction dataframes
    for track_id, group in df.groupby('track_id', sort=False):
        context = group[group['role'] == 'context']
        pred = group[group['role'] == 'prediction']
        if len(context) == 0 or len(pred) == 0:
            continue
        #yield to avoid loading all tracks into memory at once
        if with_track_id:
            yield track_id, context, pred 
        else:
            yield context, pred


def load_tracks_cached_numpy(data_dir, split='test', context_filter=None, sample_pct=100, seed=42):
    #loads all tracks for the specified split and context into RAM as numpy arrays,
    #massively reduces time spent in I/O
    files = _resolve_files(data_dir, split, context_filter)
    files = subsample_items(files, sample_pct=sample_pct, seed=seed)
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
            dx_norm_ctx = ctx['dx_norm'].to_numpy(dtype=float, copy=True)
            dy_norm_ctx = ctx['dy_norm'].to_numpy(dtype=float, copy=True)
            sog_norm_ctx = ctx['sog_norm'].to_numpy(dtype=float, copy=True)
            cog_sin_norm_ctx = ctx['cog_sin_norm'].to_numpy(dtype=float, copy=True)
            cog_cos_norm_ctx = ctx['cog_cos_norm'].to_numpy(dtype=float, copy=True)
            dt_norm_ctx = ctx['dt_norm'].to_numpy(dtype=float, copy=True)
            rot_norm_ctx = ctx['rot_norm'].to_numpy(dtype=float, copy=True)
            true_xy = pred[['x', 'y']].to_numpy(dtype=float, copy=True)

            cached_tracks.append({
                'x_ctx': x_ctx,
                'y_ctx': y_ctx,
                'rot_ctx': rot_ctx,
                'dx_norm_ctx': dx_norm_ctx,
                'dy_norm_ctx': dy_norm_ctx,
                'sog_norm_ctx': sog_norm_ctx,
                'cog_sin_norm_ctx': cog_sin_norm_ctx,
                'cog_cos_norm_ctx': cog_cos_norm_ctx,
                'dt_norm_ctx': dt_norm_ctx,
                'rot_norm_ctx': rot_norm_ctx,
                'last_x': float(x_ctx[-1]),
                'last_y': float(y_ctx[-1]),
                'true_xy': true_xy,
                'n_pred_steps': int(len(true_xy)),
            })

    print(f"Cached {len(cached_tracks)} numpy track(s) in memory")
    return cached_tracks


def reconstruct_positions(last_x, last_y, displacements):
    # models predict displacements (dx, dy) 
    # need absolute positions for evaluation
    # example: last_x = 100, last_y = 200, displacements = [[10, 0], [10, 0], [10, 0]]

    positions = np.cumsum(displacements, axis=0) #[[10, 0], [20, 0], [30, 0]]
    positions[:, 0] += last_x
    positions[:, 1] += last_y
    return positions # [[110, 200], [120, 200], [130, 200]]


def _default_model_key(model):
    name = getattr(model, 'name', 'model')
    key = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
    return key if key else 'model'

def export_predictions_for_file(model, file_path, output_dir, split='test', model_key=None, model_label=None):
    # runs prediction on test dataset
    # exports CSV 
    
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
        'dx_pred_q10',
        'dy_pred_q10',
        'dx_pred_q50',
        'dy_pred_q50',
        'dx_pred_q90',
        'dy_pred_q90',
        'x_pred',
        'y_pred',
        'x_pred_q10',
        'y_pred_q10',
        'x_pred_q50',
        'y_pred_q50',
        'x_pred_q90',
        'y_pred_q90',
        'x_gt',
        'y_gt',
    ]

    row_count = 0
    df = _read_track_file(file_path)

    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for track_id, context_df, pred_df in _iter_track_groups(df, with_track_id=True):
            n_pred_steps = len(pred_df)
            last_x = context_df['x'].iloc[-1]
            last_y = context_df['y'].iloc[-1]

            quantile_predictions = model.predict_quantiles(context_df, n_pred_steps) if hasattr(model, 'predict_quantiles') else None
            try:
                displacements = np.asarray(quantile_predictions[0.5], dtype=float)
            except Exception:
                displacements = model.predict(context_df, n_pred_steps)
            pred_positions = reconstruct_positions(last_x, last_y, displacements)
            lower_positions = reconstruct_positions(last_x, last_y, quantile_predictions[0.1]) if quantile_predictions is not None else None
            median_positions = pred_positions
            upper_positions = reconstruct_positions(last_x, last_y, quantile_predictions[0.9]) if quantile_predictions is not None else None

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
                    'dx_pred_q10': float(quantile_predictions[0.1][i, 0]) if quantile_predictions is not None else '',
                    'dy_pred_q10': float(quantile_predictions[0.1][i, 1]) if quantile_predictions is not None else '',
                    'dx_pred_q50': float(displacements[i, 0]),
                    'dy_pred_q50': float(displacements[i, 1]),
                    'dx_pred_q90': float(quantile_predictions[0.9][i, 0]) if quantile_predictions is not None else '',
                    'dy_pred_q90': float(quantile_predictions[0.9][i, 1]) if quantile_predictions is not None else '',
                    'x_pred': float(pred_positions[i, 0]),
                    'y_pred': float(pred_positions[i, 1]),
                    'x_pred_q10': float(lower_positions[i, 0]) if lower_positions is not None else '',
                    'y_pred_q10': float(lower_positions[i, 1]) if lower_positions is not None else '',
                    'x_pred_q50': float(median_positions[i, 0]),
                    'y_pred_q50': float(median_positions[i, 1]),
                    'x_pred_q90': float(upper_positions[i, 0]) if upper_positions is not None else '',
                    'y_pred_q90': float(upper_positions[i, 1]) if upper_positions is not None else '',
                    'x_gt': float(pred_df['x'].iloc[i]),
                    'y_gt': float(pred_df['y'].iloc[i]),
                })
                row_count += 1

    return {'output_path': output_path, 'rows': row_count}


def _evaluate_tracks(model, tracks):
    # evaluates a list of (context_df, pred_df) pairs for a single file,
    # returns metrics and arrays of pred positions and ground truth positions
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


def evaluate_model_cached_numpy(model, cached_tracks):
    # loop that evaluates a model on all pre-loaded tracks
    # returns metrics

    all_true = []
    all_pred = []

    batch_size = getattr(model, 'batch_size', 1)
    if hasattr(model, 'predict_batch_from_cached_tracks') and batch_size > 1:
        valid_tracks = [t for t in cached_tracks if t['n_pred_steps'] > 0]
        for i in range(0, len(valid_tracks), batch_size):
            batch = valid_tracks[i:i + batch_size]
            n_pred_steps = batch[0]['n_pred_steps']
            batch_displacements = model.predict_batch_from_cached_tracks(batch, n_pred_steps)
            for track, displacements in zip(batch, batch_displacements):
                pred_positions = reconstruct_positions(track['last_x'], track['last_y'], displacements)
                true_positions = track['true_xy']
                if len(pred_positions) != len(true_positions):
                    continue
                all_true.append(true_positions)
                all_pred.append(pred_positions)
    else:
        for track in cached_tracks:
            n_pred_steps = track['n_pred_steps']
            if n_pred_steps <= 0:
                continue

            if hasattr(model, 'predict_from_cached_track'):
                displacements = model.predict_from_cached_track(track, n_pred_steps)
            else:
                displacements = model.predict_from_arrays(
                    track['x_ctx'],
                    track['y_ctx'],
                    track['rot_ctx'],
                    n_pred_steps,
                )
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


def plot_horizon_error(metrics, label=None, ax=None, step_duration_s=30, save_path=None):
    # wrapper: pass the dict returned by evaluate_model/run_evaluation
    return plot_ade_over_horizon(
        metrics['ADE_per_step'],
        step_duration_s=step_duration_s,
        label=label,
        ax=ax,
        save_path=save_path,
    )



