import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# --- TRAIN/TEST SPLIT STRATEGY ---
# Note: To avoid data leakage, you should split 'input_files' or unique 'vessel_ids'
# into two lists (train_list, test_list) before processing. 
# Never split the resulting 'track_id's randomly!

# --- CONFIGURATION ---
WINDOW_DUR = timedelta(minutes=10)
PRED_HORIZON = timedelta(minutes=2)
STRIDE_DUR = timedelta(minutes=2)  # Set to 10s for original behavior, 2m+ for efficiency
MIN_SOG = 1.0 
LOW_SPEED_THRESH = 2.0 
LOW_SPEED_FRAC_MAX = 0.3

def infer_freq_from_name(name: str) -> int:
    """Infer sampling frequency from filename."""
    lower = name.lower()
    if 'kiel' in lower or 'bremerhaven' in lower:
        return 10
    elif 'marinecadastre' in lower or 'mississippi' in lower:
        return 60
    return 10

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
        # Time Gap Guard: Ensure the window doesn't span a data blackout
        # The total window (12m) should not realistically take more than, say, 15m of wall time.
        start_t = group.iloc[i - n_ctx]['t_utc']
        end_t = group.iloc[i + n_pred - 1]['t_utc']
        if (end_t - start_t) > (WINDOW_DUR + PRED_HORIZON) * 1.5:
            continue

        # Extract slices
        ctx = group.iloc[i - n_ctx : i].copy()
        pred = group.iloc[i : i + n_pred].copy()
        
        # Label roles
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
    # 2. Less than 30% of points can be < 2.0 knots
    valid_mask = (stats['max_sog'] >= MIN_SOG) & (stats['low_speed_frac'] <= LOW_SPEED_FRAC_MAX)
    valid_ids = stats.index[valid_mask]

    removed_anchoring = (stats['max_sog'] < MIN_SOG).sum()
    removed_lowspeed = (stats['low_speed_frac'] > LOW_SPEED_FRAC_MAX).sum()
    print(f"Speed filter: {len(stats)} total, removed anchoring: {removed_anchoring}, low-speed: {removed_lowspeed}, kept: {len(valid_ids)}")

    return df[df['track_id'].isin(valid_ids)].copy()

if __name__ == '__main__':
    input_dir = Path('output/03_sampled')
    output_dir = Path('output/04_trajectories')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(input_dir.glob('*.csv'))
    global_track_counter = 0
    
    for src in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {src.name}")
        print(f"{'='*60}")
        df = pd.read_csv(src, parse_dates=['t_utc'])
        freq_s = infer_freq_from_name(src.name)
        
        # Generate ALL segments first
        all_segments = []
        vessel_groups = df.groupby('vessel_id')
        
        for _, v_data in vessel_groups:
            segments, global_track_counter = create_samples_efficient(
                v_data, freq_s, global_track_counter
            )
            all_segments.extend(segments)
        
        if not all_segments:
            print(f"No valid segments for {src.name}")
            continue
        
        # Combine + speed filter FIRST
        processed_df = pd.concat(all_segments, ignore_index=True)
        filtered_df = apply_speed_filters(processed_df)
        
        if len(filtered_df) == 0:
            print(f"No valid tracks after filtering for {src.name}")
            continue
        
        n_total_tracks = filtered_df['track_id'].nunique()
        print(f"Generated {n_total_tracks:,} tracks after speed filtering")
        
        # === NOW SPLIT TRACKS 60/20/20 ===
        np.random.seed(42)
        all_tracks = filtered_df['track_id'].unique()
        np.random.shuffle(all_tracks)
        
        split_train = int(0.60 * len(all_tracks))
        split_val   = int(0.80 * len(all_tracks))
        
        tracks_train = set(all_tracks[:split_train])
        tracks_val   = set(all_tracks[split_train:split_val])
        tracks_test  = set(all_tracks[split_val:])
        
        print(f"Track split (post-filter): {len(tracks_train):,} train | "
              f"{len(tracks_val):,} val | {len(tracks_test):,} test "
              f"(total {n_total_tracks:,} tracks)")
        
        # Save splits
        for split_name, track_set in [("train", tracks_train), 
                                      ("val", tracks_val), 
                                      ("test", tracks_test)]:
            split_df = filtered_df[filtered_df['track_id'].isin(track_set)]
            dst = output_dir / f"{split_name}_{src.name}"
            split_df.to_csv(dst, index=False)
            print(f"  Saved {len(track_set):,} {split_name} tracks → {dst}")
               
        print(f"{src.name} complete")
