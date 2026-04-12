# Slicing trajectories into samples 
# Input:  output/03_sampled/*.csv
# Output: output/04_trajectories/*.csv

#On full dataset: Final Report: 86476 segments were too short after context-splitting and were discarded.

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# To avoid data leakage, we split unique 'vessel_ids'. See thesis text

WINDOW_DUR = timedelta(minutes=5)
PRED_HORIZON = timedelta(minutes=5)
STRIDE_DUR_DEFAULT = timedelta(minutes=1)  
STRIDE_DUR_LOCK = timedelta(seconds=10)    # smaller for locks
MIN_SOG = 0.5 
LOW_SPEED_THRESH = 1.0 
LOW_SPEED_FRAC_MAX = 0.5
AUGMENT: bool = True  # toggle for data augmentation
GERMAN_FREQ_S = 30

CONTEXT_MAPPING = {
    'river': 0, 
    'channel': 1, 
    'harbour': 2, 
    'lock': 3, 
    'unknown': 4
}

def create_samples_efficient(group: pd.DataFrame, freq_s: int, start_track_id: int, stride_s: int):
    #Slices vessel trajectory into context/prediction using stride
    group = group.sort_values('t_utc').reset_index(drop=True)
    
    n_ctx = int(WINDOW_DUR.total_seconds() / freq_s)
    n_pred = int(PRED_HORIZON.total_seconds() / freq_s)
    n_stride = max(1, int(stride_s / freq_s)) 
    
    segments = []
    current_id = start_track_id
    
    for i in range(n_ctx, len(group) - n_pred, n_stride):
        # Time Gap Guard: Ensure the window doesn't span a data blackout, see thesis text
        #start_t = group.iloc[i - n_ctx]['t_utc']
        #end_t = group.iloc[i + n_pred - 1]['t_utc']
        start_t = group['t_utc'].iat[i - n_ctx] # direkt access is faster than iloc for single values
        end_t = group['t_utc'].iat[i + n_pred - 1]
        if (end_t - start_t) > (WINDOW_DUR + PRED_HORIZON) * 1.1:  # Allow 10% tolerance for irregular sampling. 
            #Being stricter here will make assumpotions in the model about regular sampling more valid, but will throw out more segments. Adjust as needed.
            continue

        ctx = group.iloc[i - n_ctx : i].copy()
        pred = group.iloc[i : i + n_pred].copy()
        
        ctx['role'] = 'context'
        pred['role'] = 'prediction'
        
        combined = pd.concat([ctx, pred])
        combined['track_id'] = current_id
        
        segments.append(combined)
        current_id += 1
        
    return segments, current_id

def apply_speed_filters(df: pd.DataFrame) -> pd.DataFrame:
    #Vectorized filtering of segments based on speed criteria
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
    # 2. Less than LOW_SPEED_FRAC_MAX of points can be < 2.0 knots (MIN_SOG)
    valid_mask = (stats['max_sog'] >= MIN_SOG) & (stats['low_speed_frac'] <= LOW_SPEED_FRAC_MAX)
    valid_ids = stats.index[valid_mask]

    removed_anchoring = (stats['max_sog'] < MIN_SOG).sum()
    removed_lowspeed = (stats['low_speed_frac'] > LOW_SPEED_FRAC_MAX).sum()
    #print(f"Speed filter: {len(stats)} total, removed anchoring: {removed_anchoring}, low-speed: {removed_lowspeed}, kept: {len(valid_ids)}")

    return df[df['track_id'].isin(valid_ids)].copy()

def smooth_context_labels(context_series, window_size=6):
    #Applies rolling majority vote to labels to remove GNSS jitter
    int_to_label = {v: k for k, v in CONTEXT_MAPPING.items()}
    
    numeric_series = context_series.map(CONTEXT_MAPPING)

    # Pure numpy mode calculation
    def get_mode(x):
        values, counts = np.unique(x, return_counts=True) 
        return values[np.argmax(counts)]
    
    # raw=True passes raw numpy arrays instead of Pandas objects, should be faster
    smoothed_numeric = numeric_series.rolling(window=window_size, min_periods=1).apply(
        get_mode, raw=True
    )
    return smoothed_numeric.map(int_to_label)

def augment_rotate(df: pd.DataFrame, angle_deg: float, new_track_id_start: int) -> pd.DataFrame:
    df_aug = df.copy()
        
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    x = df_aug['x']
    y = df_aug['y']
    
    # centroid of current trajectory
    cx = x.mean()
    cy = y.mean()
    
    # 1. Shift to origin (x - cx, y - cy)
    # 2. Rotate (* c, * s)
    # 3. Shift back to original centroid (+ cx, + cy)
    df_aug['x'] = (x - cx) * c - (y - cy) * s + cx
    df_aug['y'] = (x - cx) * s + (y - cy) * c + cy
    
    # Adjust angles by adding the rotation angle modulo 360
    if 'cog' in df_aug.columns:
        df_aug['cog'] = (df_aug['cog'] + angle_deg) % 360
    if 'heading' in df_aug.columns:
        df_aug['heading'] = (df_aug['heading'] + angle_deg) % 360
    if 'true_heading' in df_aug.columns: # optional feature
        df_aug['true_heading'] = (df_aug['true_heading'] + angle_deg) % 360   

    # we might want to add an identifyer to the track_id to indicate augmentation, for now just re-assign new unique track IDs
    unique_ids = df_aug['track_id'].unique() 
    track_mapping = {old_id: new_id for old_id, new_id in zip(unique_ids, range(new_track_id_start, new_track_id_start + len(unique_ids)))}
    df_aug['track_id'] = df_aug['track_id'].map(track_mapping)
    
    return df_aug

def augment_mirror(df: pd.DataFrame, new_track_id_start: int) -> pd.DataFrame:
    df_aug = df.copy()
    
    cx = df_aug['x'].mean()
    
    # Mirror across local North-South axis (Y-axis) at the centroid
    df_aug['x'] = cx - (df_aug['x'] - cx)
    
    # Adjust angles 
    df_aug['cog'] = (360 - df_aug['cog']) % 360
    df_aug['heading'] = (360 - df_aug['heading']) % 360

    # note rn rot is nan and calculated later, but this is a safeguard for any future changes
    if 'rot' in df_aug.columns:
        df_aug['rot'] = -df_aug['rot']


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
        print(f"Processing: {src.name}")
        df = pd.read_csv(src, parse_dates=['t_utc'])
        freq_s = GERMAN_FREQ_S

        #only necessary for runs before rerunning 01_extract on old test files that have NaNs in sog and cog. 
        #now  01_extract already removes these rows
        #n_before = len(df)
        #df = df.dropna(subset=['sog', 'rot', 'context']).reset_index(drop=True)
        if df.empty:
            continue


        # stratification features:
        vessel_profiles = df.groupby('vessel_id').agg({
            'sog': 'mean', # use mean SOG
            'rot': lambda x: x.abs().max(), # use max rot
            'context': lambda x: x.mode()[0] # use most frequent context
        }).reset_index()

        # Convert context strings to numbers 
        vessel_profiles['context_idx'] = vessel_profiles['context'].map(CONTEXT_MAPPING)

        # Scale features: KMeans needs this so ROT (0-90) doesn't dominate SOG (0-15)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(vessel_profiles[['sog', 'rot', 'context_idx']])
        
        # Cluster vessels into 5 Strata based on (SOG, ROT, Context)
        n_clusters = min(5, len(vessel_profiles))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        vessel_profiles['stratum'] = km.fit_predict(scaled_features)

        cluster_counts = vessel_profiles['stratum'].value_counts().sort_index()

        small_clusters = cluster_counts[cluster_counts < 6].index # avoid issue with wedel dataset where some clusters have only 1 vessel, which causes errors in stratified splitting.
        if not small_clusters.empty:
            largest_cluster = cluster_counts.idxmax()
            vessel_profiles.loc[vessel_profiles['stratum'].isin(small_clusters), 'stratum'] = largest_cluster # simply assign to largest cluster

        # 70/15/15 Stratified Split
        v_train, v_temp = train_test_split(vessel_profiles['vessel_id'], test_size=0.30, 
                                           stratify=vessel_profiles['stratum'], random_state=42)
        
        temp_profiles = vessel_profiles[vessel_profiles['vessel_id'].isin(v_temp)]
        v_val, v_test = train_test_split(temp_profiles['vessel_id'], test_size=0.50, 
                                         stratify=temp_profiles['stratum'], random_state=42)

        vessels_train, vessels_val = set(v_train), set(v_val)

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
                continue
            
            split_df = pd.concat(segments, ignore_index=True)
            
            # Group by context before filtering and saving
            for context_label, context_df in split_df.groupby('context'):
                if context_label == 'unknown':
                    continue
                filtered_df = apply_speed_filters(context_df)
                
                if AUGMENT and split_name == 'train' and context_label == 'lock' and not filtered_df.empty:
                    print(f" Augmenting {context_label} ")
                    
                    augmented_dfs = [filtered_df]
                    
                    random_angles = np.random.uniform(0, 360, size=16)
                    
                    for angle in random_angles:
                        # Rotate
                        rotated_df = augment_rotate(filtered_df, angle_deg=angle, new_track_id_start=global_track_counter)
                        global_track_counter += rotated_df['track_id'].nunique()
                        augmented_dfs.append(rotated_df)
                        
                        # mirror the rotated dataset
                        mirrored_df = augment_mirror(rotated_df, new_track_id_start=global_track_counter)
                        global_track_counter += mirrored_df['track_id'].nunique()
                        augmented_dfs.append(mirrored_df)
                    
                    filtered_df = pd.concat(augmented_dfs, ignore_index=True)

                if not filtered_df.empty:
                    dst = output_dir / f"{split_name}_{context_label}_{src.name}"
                    filtered_df.to_csv(dst, index=False)
                    n_tracks = filtered_df['track_id'].nunique()
                    print(f" Saved {n_tracks:,} tracks to {dst.name}")
            
    print(f"\n {total_short_cuts} segments were too short after context-splitting and were discarded")
    print(f"Finished segmenting all files")