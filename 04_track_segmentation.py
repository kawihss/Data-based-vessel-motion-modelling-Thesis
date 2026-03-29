# 04_track_segmentation.py: Slicing trajectories into samples based on time and context
# Input:  output/03_sampled/*.csv
# Output: output/04_trajectories/*.csv

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# To avoid data leakage, we split unique 'vessel_ids'. See thesis text

# --- CONFIGURATION ---
WINDOW_DUR = timedelta(minutes=5)
PRED_HORIZON = timedelta(minutes=5)
STRIDE_DUR = timedelta(minutes=1)  # how much to slide the window for the next sample, smaller means more samples but more overlap. Adjust to get -150-200k Samples per domain
MIN_SOG = 0.5 
LOW_SPEED_THRESH = 1.0 
LOW_SPEED_FRAC_MAX = 0.5

def infer_freq_from_name(name: str) -> int:
    #"""Infer sampling frequency from filename."""
    #lower = name.lower()
    #if 'kiel' in lower or 'bremerhaven' in lower or 'wedel' in lower:
     #   return 30
    #elif 'marinecadastre' in lower or 'mississippi' in lower:
    #   return 60
    return 30

def create_samples_efficient(group: pd.DataFrame, freq_s: int, start_track_id: int):
    """
    Slices the vessel trajectory into context/prediction pairs using a stride.
    """
    group = group.sort_values('t_utc').reset_index(drop=True)
    
    # Calculate how many rows represent our durations
    n_ctx = int(WINDOW_DUR.total_seconds() / freq_s)
    n_pred = int(PRED_HORIZON.total_seconds() / freq_s)
    n_stride = max(1, int(STRIDE_DUR.total_seconds() / freq_s))
    
    segments = []
    current_id = start_track_id
    
    # Slide the window using the stride
    for i in range(n_ctx, len(group) - n_pred, n_stride):
        # Time Gap Guard: Ensure the window doesn't span a data blackout, see thesis text
        start_t = group.iloc[i - n_ctx]['t_utc']
        end_t = group.iloc[i + n_pred - 1]['t_utc']
        if (end_t - start_t) > (WINDOW_DUR + PRED_HORIZON) * 1.1:  # Allow 10% tolerance for irregular sampling
            continue

        # context prediction split
        ctx = group.iloc[i - n_ctx : i].copy()
        pred = group.iloc[i : i + n_pred].copy()
        
        ctx['role'] = 'context'
        pred['role'] = 'prediction'
        
        # Combine and assign unique ID
        combined = pd.concat([ctx, pred])
        combined['track_id'] = current_id
        
        segments.append(combined)
        current_id += 1
        
    return segments, current_id

def apply_speed_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized filtering of segments based on speed criteria."""
    if df.empty:
        return df
    
    # Calculate metrics for every track_id at once
    stats = df.groupby('track_id')['sog'].agg(
        max_sog='max',
        low_speed_count=lambda s: (s < LOW_SPEED_THRESH).sum(),
        total_points='count'
    )
    
    stats['low_speed_frac'] = stats['low_speed_count'] / stats['total_points']
    
    # Filter IDs based on criteria:
    # 1. Max SOG must be >= 1.0 (removes anchored ships)
    # 2. Less than LOW_SPEED_FRAC_MAX of points can be < 2.0 knots
    valid_mask = (stats['max_sog'] >= MIN_SOG) & (stats['low_speed_frac'] <= LOW_SPEED_FRAC_MAX)
    valid_ids = stats.index[valid_mask]

    removed_anchoring = (stats['max_sog'] < MIN_SOG).sum()
    removed_lowspeed = (stats['low_speed_frac'] > LOW_SPEED_FRAC_MAX).sum()
    print(f"Speed filter: {len(stats)} total, removed anchoring: {removed_anchoring}, low-speed: {removed_lowspeed}, kept: {len(valid_ids)}")

    return df[df['track_id'].isin(valid_ids)].copy()

def smooth_context_labels(context_series, window_size=6):
    """
    Applies a rolling majority vote to string labels to remove GNSS jitter.
    Pandas rolling() requires numeric data, so we temporarily map strings to integers.
    """
    #Create mapping dictionaries
    unique_labels = context_series.unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    
    # Map string labels to integers for rolling operation
    numeric_series = context_series.map(label_to_int)
    
    # Apply rolling window to find the most common integer (mode)
    smoothed_numeric = numeric_series.rolling(window=window_size, min_periods=1).apply(
        lambda x: pd.Series(x).mode()[0]
    )
    
    # Map back to original string labels
    return smoothed_numeric.map(int_to_label)

if __name__ == '__main__':
    input_dir = Path('output/03_sampled')
    output_dir = Path('output/04_trajectories')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(input_dir.glob('*.csv'))
    global_track_counter = 0
    
    total_short_cuts = 0

    for src in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {src.name}")
        print(f"{'='*60}")
        df = pd.read_csv(src, parse_dates=['t_utc'])
        freq_s = infer_freq_from_name(src.name)
        
        #Split vessels into train/val/test
        np.random.seed(42)
        all_vessels = df['vessel_id'].unique()
        np.random.shuffle(all_vessels)
        
        n_vessels = len(all_vessels)
        split_train = int(0.70 * n_vessels)
        split_val   = int(0.85 * n_vessels)
        
        vessels_train = set(all_vessels[:split_train])
        vessels_val   = set(all_vessels[split_train:split_val])
        
        split_segments = {"train": [], "val": [], "test": []}
        
        #Generate Segments: Split by vessel AND context change
        vessel_groups = df.groupby('vessel_id')
        min_pts_required = int((WINDOW_DUR + PRED_HORIZON).total_seconds() / freq_s)

        for v_id, v_data in vessel_groups:
            v_data = v_data.sort_values('t_utc')
            
            # Label stability fix (2-min rolling mode)
            # Note: Complex mapping required because pandas lacks a native rolling string mode.
            c_map = {c: i for i, c in enumerate(v_data['context'].unique())}
            #v_data['context'] = v_data['context'].map(c_map).rolling(6, min_periods=1).apply(
            #    lambda x: pd.Series(x).mode()[0]).map({i: c for c, i in c_map.items()})
            # Label stability fix (rolling mode to smooth GNSS jitter)
            v_data['context'] = smooth_context_labels(v_data['context'], window_size=6)


            # Identifies where context changes within the same vessel
            context_changed = v_data['context'] != v_data['context'].shift(1)
            v_data['context_group'] = context_changed.cumsum()
            
            for _, sub_data in v_data.groupby('context_group'):
                if len(sub_data) < min_pts_required:
                    total_short_cuts += 1
                    continue
                
                segments, global_track_counter = create_samples_efficient(
                    sub_data, freq_s, global_track_counter
                )
                
                # Assign to splits
                target = "test"
                if v_id in vessels_train: target = "train"
                elif v_id in vessels_val: target = "val"
                split_segments[target].extend(segments)

        #Process each split: apply speed filters and save
        for split_name, segments in split_segments.items():
            if not segments:
                print(f" ! No segments generated for {split_name} split in {src.name}")
                continue
            
            split_df = pd.concat(segments, ignore_index=True)
            
            # Group by context before filtering and saving
            for context_label, context_df in split_df.groupby('context'):
                if context_label == 'unknown':
                    continue
                print(f"Processing {split_name} split - Context: {context_label}")
                filtered_df = apply_speed_filters(context_df)
                
                if not filtered_df.empty:
                    dst = output_dir / f"{split_name}_{context_label}_{src.name}"
                    filtered_df.to_csv(dst, index=False)
                    n_tracks = filtered_df['track_id'].nunique()
                    print(f" Saved {n_tracks:,} tracks to {dst.name}")
            
    print(f"\nFinal Report: {total_short_cuts} segments were too short after context-splitting and were discarded.")
    print(f"Finished processing all files.")