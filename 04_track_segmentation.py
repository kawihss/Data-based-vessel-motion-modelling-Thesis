# 04_track_segmentation.py: Slicing trajectories into samples based on time and context
# Input:  output/03_sampled/*.csv
# Output: output/04_trajectories/*.csv

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# To avoid data leakage, we split unique 'vessel_ids'. See thesis text

# --- CONFIGURATION ---
WINDOW_DUR = timedelta(minutes=5)
PRED_HORIZON = timedelta(minutes=5)
STRIDE_DUR_DEFAULT = timedelta(minutes=1)  
STRIDE_DUR_LOCK = timedelta(seconds=10)    # smaller for locks
MIN_SOG = 0.5 
LOW_SPEED_THRESH = 1.0 
LOW_SPEED_FRAC_MAX = 0.5
AUGMENT: bool = True  # Global toggle for data augmentation

def infer_freq_from_name(name: str) -> int:
    return 30

def create_samples_efficient(group: pd.DataFrame, freq_s: int, start_track_id: int, stride_s: int):
    """
    Slices the vessel trajectory into context/prediction pairs using a stride.
    """
    group = group.sort_values('t_utc').reset_index(drop=True)
    
    # Calculate how many rows represent our durations
    n_ctx = int(WINDOW_DUR.total_seconds() / freq_s)
    n_pred = int(PRED_HORIZON.total_seconds() / freq_s)
    n_stride = max(1, int(stride_s / freq_s)) 
    
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

def augment_rotate(df: pd.DataFrame, angle_deg: float, new_track_id_start: int) -> pd.DataFrame:
    df_aug = df.copy()
        
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    x = df_aug['x']
    y = df_aug['y']
    
    # turn around center of current trajectory
    cx = x.mean()
    cy = y.mean()
    
    # 1. Shift to origin (x - cx, y - cy)
    # 2. Rotate (* c, * s)
    # 3. Shift back to original centroid (+ cx, + cy)
    df_aug['x'] = (x - cx) * c - (y - cy) * s + cx
    df_aug['y'] = (x - cx) * s + (y - cy) * c + cy
    
    # Adjust angles (COG and Heading) by adding the rotation angle modulo 360
    if 'cog' in df_aug.columns:
        df_aug['cog'] = (df_aug['cog'] + angle_deg) % 360
    if 'heading' in df_aug.columns:
        df_aug['heading'] = (df_aug['heading'] + angle_deg) % 360
    if 'true_heading' in df_aug.columns: # optional feature
        df_aug['true_heading'] = (df_aug['true_heading'] + angle_deg) % 360   


    # Re-assign track IDs to ensure global uniqueness
    unique_ids = df_aug['track_id'].unique()
    track_mapping = {old_id: new_id for old_id, new_id in zip(unique_ids, range(new_track_id_start, new_track_id_start + len(unique_ids)))}
    df_aug['track_id'] = df_aug['track_id'].map(track_mapping)
    
    return df_aug

def augment_mirror(df: pd.DataFrame, new_track_id_start: int) -> pd.DataFrame:
    df_aug = df.copy()
    
    cx = df_aug['x'].mean()
    
    # Mirror across the local North-South axis (Y-axis) at the centroid
    df_aug['x'] = cx - (df_aug['x'] - cx)
    
    # Adjust angles 
    df_aug['cog'] = (360 - df_aug['cog']) % 360
    df_aug['heading'] = (360 - df_aug['heading']) % 360

    # Mirroring reverses the turn direction (Right turn becomes Left turn)
    # note rn rot is nan and calculated later, but this is a safeguard for any future changes
    if 'rot' in df_aug.columns:
        df_aug['rot'] = -df_aug['rot']


    # Re-assign track IDs to ensure global uniqueness
    unique_ids = df_aug['track_id'].unique()
    track_mapping = {old_id: new_id for old_id, new_id in zip(unique_ids, range(new_track_id_start, new_track_id_start + len(unique_ids)))}
    df_aug['track_id'] = df_aug['track_id'].map(track_mapping)
    
    return df_aug

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

        # strafication:
        vessel_profiles = df.groupby('vessel_id').agg({
            'sog': 'mean', # use mean SOG
            'rot': lambda x: x.abs().max(), # use max rot
            'context': lambda x: x.mode()[0] # use most frequent context
        }).reset_index()

        # Convert context strings to numbers for the algorithm
        vessel_profiles['context_idx'] = vessel_profiles['context'].astype('category').cat.codes

        # Scale features: KMeans needs this so ROT (0-90) doesn't dominate SOG (0-15)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(vessel_profiles[['sog', 'rot', 'context_idx']])
        
        # Cluster vessels into 8 Strata based on (SOG, ROT, Context)
        n_clusters = min(8, len(vessel_profiles))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        vessel_profiles['stratum'] = km.fit_predict(scaled_features)

        print(f"Stratification Summary for {src.name} ---")
        cluster_counts = vessel_profiles['stratum'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} vessels")

        # 70/15/15 Stratified Split
        v_train, v_temp = train_test_split(vessel_profiles['vessel_id'], test_size=0.30, 
                                           stratify=vessel_profiles['stratum'], random_state=42)
        
        temp_profiles = vessel_profiles[vessel_profiles['vessel_id'].isin(v_temp)]
        v_val, v_test = train_test_split(temp_profiles['vessel_id'], test_size=0.50, 
                                         stratify=temp_profiles['stratum'], random_state=42)

        vessels_train, vessels_val = set(v_train), set(v_val)
        #print(f"Split Distribution:")
        #print(f"  Train: {len(vessels_train)} vessels")
        #print(f"  Val:   {len(vessels_val)} vessels")
        #print(f"  Test:  {len(v_test)} vessels")
        split_segments = {"train": [], "val": [], "test": []}
        
        #Generate Segments: Split by vessel AND context change
        vessel_groups = df.groupby('vessel_id')
        min_pts_required = int((WINDOW_DUR + PRED_HORIZON).total_seconds() / freq_s)

        for v_id, v_data in vessel_groups:
            v_data = v_data.sort_values('t_utc')
            
            v_data['context'] = smooth_context_labels(v_data['context'], window_size=6)

            # Identifies where context changes within the same vessel
            context_changed = v_data['context'] != v_data['context'].shift(1)
            v_data['context_group'] = context_changed.cumsum()
            
            for _, sub_data in v_data.groupby('context_group'):
                if len(sub_data) < min_pts_required:
                    total_short_cuts += 1
                    continue
                
                current_context = sub_data['context'].iloc[0]
                if current_context == 'lock':
                    current_stride_s = int(STRIDE_DUR_LOCK.total_seconds())
                else:
                    current_stride_s = int(STRIDE_DUR_DEFAULT.total_seconds())
                
                segments, global_track_counter = create_samples_efficient(
                    sub_data, freq_s, global_track_counter, current_stride_s
                )
                
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
                
                if AUGMENT and split_name == 'train' and context_label == 'lock' and not filtered_df.empty:
                    print(f" Augmenting {context_label} data in train split...")
                    
                    augmented_dfs = [filtered_df]
                    
                    random_angles = np.random.uniform(0, 360, size=16)
                    
                    for angle in random_angles:
                        # 1. Rotate
                        rotated_df = augment_rotate(filtered_df, angle_deg=angle, new_track_id_start=global_track_counter)
                        global_track_counter += rotated_df['track_id'].nunique()
                        augmented_dfs.append(rotated_df)
                        
                        # 2. mirror the rotated dataset
                        mirrored_df = augment_mirror(rotated_df, new_track_id_start=global_track_counter)
                        global_track_counter += mirrored_df['track_id'].nunique()
                        augmented_dfs.append(mirrored_df)
                    
                    filtered_df = pd.concat(augmented_dfs, ignore_index=True)

                if not filtered_df.empty:
                    dst = output_dir / f"{split_name}_{context_label}_{src.name}"
                    filtered_df.to_csv(dst, index=False)
                    n_tracks = filtered_df['track_id'].nunique()
                    print(f" Saved {n_tracks:,} tracks to {dst.name}")
            
    print(f"\nFinal Report: {total_short_cuts} segments were too short after context-splitting and were discarded.")
    print(f"Finished processing all files.")