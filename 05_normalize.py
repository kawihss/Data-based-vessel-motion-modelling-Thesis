# 05_normalize.py — Normalization: displacements, sin/cos COG, z-score
# split is done before normalization and normalization is applied per split, 
# scaler is fitted on train only and applied to all splits to avoid data leakage
# Input:  output/04_trajectories/*.csv 
# Output: output/05_normalized/*.csv + scalers.pkl

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ['dx', 'dy', 'sog', 'cog_sin', 'cog_cos', 'dt']


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dx/dy displacements and sin/cos COG encoding across all tracks."""
    df = df.copy().sort_values(['track_id', 't_utc'])

    df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
    df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

    cog_rad = np.deg2rad(df['cog'].fillna(360))
    df['cog_sin'] = np.sin(cog_rad)
    df['cog_cos'] = np.cos(cog_rad)

    return df


def normalize_dataset(df: pd.DataFrame, scaler=None, fit=False, verbose=True):
    """Compute features per track, then z-score."""

    df = compute_features(df)

    if fit:
        scaler = StandardScaler()
        scaler.fit(df[NUMERIC_FEATURES])
        print(" Scaler fitted:")
        for feature, mu, sigma in zip(NUMERIC_FEATURES, scaler.mean_, scaler.scale_):
            print(f"    {feature}: μ={mu:.4f}, σ={sigma:.4f}")

    df_numeric = pd.DataFrame(
        scaler.transform(df[NUMERIC_FEATURES]),
        columns=[f'{feature}_norm' for feature in NUMERIC_FEATURES],
        index=df.index
    )

    keep_cols = ['track_id', 'role', 't_utc', 'x', 'y', 'sog', 'cog']
   
    df_out = pd.concat([df[keep_cols], df_numeric], axis=1)

    print(f" Normalized {len(df):,} rows ({df['track_id'].nunique():,} tracks)")
    return df_out, scaler


if __name__ == "__main__":
    input_dir = Path("output/04_trajectories")
    output_dir = Path("output/05_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files by dataset (train/val/test)
    train_files = sorted(input_dir.glob("train_*.csv"))
    val_files   = sorted(input_dir.glob("val_*.csv"))
    test_files  = sorted(input_dir.glob("test_*.csv"))
    
    # Fit scaler ON TRAIN ONLY
    print("Fitting scaler on TRAIN data...")
    all_train_dfs = []
    for train_file in train_files:
        df = pd.read_csv(train_file, parse_dates=['t_utc'])
        if 'window_end_t_utc' in df.columns:
            df['window_end_t_utc'] = pd.to_datetime(df['window_end_t_utc'])
        all_train_dfs.append(df)
    
    train_combined = pd.concat(all_train_dfs, ignore_index=True)
    _, scaler = normalize_dataset(train_combined, fit=True, verbose=True)
    
    print(f"Scaler fitted on {len(train_combined):,} train rows")
    
    # Process ALL splits (train/val/test) with SAME scaler
    all_files = [(f"train ({len(train_files)} files)", train_files),
                 (f"val ({len(val_files)} files)", val_files),
                 (f"test ({len(test_files)} files)", test_files)]
    
    total_rows = total_tracks = 0
    scalers = {"global_scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}}
    
    for split_name, files in all_files:
        if not files:
            print(f"No {split_name} files found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Normalizing {split_name}")
        print(f"{'='*60}")
        
        for src in files:
            df = pd.read_csv(src, parse_dates=['t_utc'])
            if 'window_end_t_utc' in df.columns:
                df['window_end_t_utc'] = pd.to_datetime(df['window_end_t_utc'])
            
            df_norm, _ = normalize_dataset(df, scaler=scaler, fit=False, verbose=False)
            
            dst = output_dir / src.name
            df_norm.to_csv(dst, index=False)
            n_rows = len(df_norm)
            n_tracks = df_norm['track_id'].nunique()
            
            print(f"  {src.name} → {n_rows:,} rows, {n_tracks:,} tracks")
            total_rows += n_rows
            total_tracks += n_tracks
        
        print(f"{split_name} complete")
    
    # Save scaler for evaluation and reproducibility
    joblib.dump(scalers, output_dir / "scalers.pkl")
    print(f"\n{'='*60}")
    print(f"NORMALIZED {total_rows:,} rows ({total_tracks:,} tracks)")
    print(f"Train-only scaler saved: {output_dir}/scalers.pkl")
    print(f"{'='*60}")
