# 05_normalize.py — Normalization: displacements, sin/cos COG, z-score
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

    # diff() per track — no apply() needed, track_id stays in df
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
        if verbose:
            print("  Scaler fitted:")
            for feat, mu, sigma in zip(NUMERIC_FEATURES, scaler.mean_, scaler.scale_):
                print(f"    {feat}: μ={mu:.4f}, σ={sigma:.4f}")

    df_numeric = pd.DataFrame(
        scaler.transform(df[NUMERIC_FEATURES]),
        columns=[f'{feat}_norm' for feat in NUMERIC_FEATURES],
        index=df.index
    )

    keep_cols = ['track_id', 'role', 't_utc', 'x', 'y', 'sog', 'cog']
    if 'window_end_t_utc' in df.columns:
        keep_cols.append('window_end_t_utc')

    df_out = pd.concat([df[keep_cols], df_numeric], axis=1)

    if verbose:
        print(f"  Normalized {len(df):,} rows ({df['track_id'].nunique():,} tracks)")
    return df_out, scaler


if __name__ == "__main__":
    input_dir  = Path("output/04_trajectories")
    output_dir = Path("output/05_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {input_dir}")
        raise SystemExit(1)

    scalers = {}
    total_rows = total_tracks = 0

    for src in input_files:
        print(f"\n{'='*60}")
        print(f"Normalizing: {src.name}")
        print(f"{'='*60}")

        df = pd.read_csv(src, parse_dates=['t_utc']).copy()
        if 'window_end_t_utc' in df.columns:
            df['window_end_t_utc'] = pd.to_datetime(df['window_end_t_utc'])
        n_rows = len(df)
        n_tracks = df['track_id'].nunique()

        df_norm, scaler = normalize_dataset(df, fit=True, verbose=True)
        scalers[src.name] = {
            'mean':  scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'features': NUMERIC_FEATURES
        }

        dst = output_dir / src.name
        df_norm.to_csv(dst, index=False)
        print(f"Saved: {dst} ({len(df_norm):,} rows, {df_norm['track_id'].nunique():,} tracks)")

        total_rows += n_rows
        total_tracks += n_tracks

    joblib.dump(scalers, output_dir / "scalers.pkl")

    print(f"\n{'='*60}")
    print(f"NORMALIZED {total_rows:,} rows ({total_tracks:,} tracks) across {len(input_files)} files")
    print(f"Scalers saved to {output_dir}/")
    print(f"{'='*60}")