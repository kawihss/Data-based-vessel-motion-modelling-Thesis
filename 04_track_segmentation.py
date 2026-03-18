import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# To avoid data leakage, we split unique 'vessel_ids'. See thesis text

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

def create_samples_efficient(group: pd.DataFrame, freq_s: int, start_track_id: int):# efficient means with overlap
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
        if (end_t - start_t) > (WINDOW_DUR + PRED_HORIZON) * 1.5:
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
        
        # 1. Split vessels into train/val/test based on unique vessel_ids to avoid data leakage, see thesis text
        np.random.seed(42)
        all_vessels = df['vessel_id'].unique()
        np.random.shuffle(all_vessels)
        
        # 2. Define splits (60% train, 20% val, 20% test)
        n_vessels = len(all_vessels)
        split_train = int(0.60 * n_vessels)
        split_val   = int(0.80 * n_vessels)
        
        vessels_train = set(all_vessels[:split_train])
        vessels_val   = set(all_vessels[split_train:split_val])
        vessels_test  = set(all_vessels[split_val:])
        
        # Containers for segments
        split_segments = {
            "train": [],
            "val": [],
            "test": []
        }
        
        # 3. Generate Segments for each vessel and assign to splits
        vessel_groups = df.groupby('vessel_id')
        for v_id, v_data in vessel_groups:
            segments, global_track_counter = create_samples_efficient(
                v_data, freq_s, global_track_counter
            )
            
            if v_id in vessels_train:
                split_segments["train"].extend(segments)
            elif v_id in vessels_val:
                split_segments["val"].extend(segments)
            else:
                split_segments["test"].extend(segments)

        # 4. Process each split: apply speed filters and save
        for split_name, segments in split_segments.items():
            if not segments:
                print(f" ! No segments generated for {split_name} split in {src.name}")
                continue
            
            # Combine all tracks for this split
            split_df = pd.concat(segments, ignore_index=True)
            
            # Apply speed filters to this split
            print(f"Processing {split_name} split")
            filtered_split_df = apply_speed_filters(split_df)
            
            if filtered_split_df.empty:
                print(f" ! No tracks remained after filtering for {split_name}")
                continue
                
            # Save to disk
            dst = output_dir / f"{split_name}_{src.name}"
            filtered_split_df.to_csv(dst, index=False)
            
            n_tracks = filtered_split_df['track_id'].nunique()
            print(f" Saved {n_tracks:,} tracks to {dst.name}")
            
        print(f"Finished processing {src.name}")