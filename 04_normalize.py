# 04_normalize.py — Normalization: displacements, sin/cos COG, z-score
# Input:  output/03_sampled/*.csv
# Output: output/04_normalized/*.csv + scalers.pkl

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler


# dt is kept as-is (time since last real point after interpolation), time since last point is constant as per sample rate
# cog is encoded as sin/cos (circular variable)
# x, y → replaced by dx, dy (displacement between consecutive points)
# sog is z-scored directly (clip outliers first)
NUMERIC_FEATURES = ['dx', 'dy', 'sog', 'cog_sin', 'cog_cos', 'dt'] 


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel: compute dx/dy displacements and sin/cos COG encoding."""
    df = df.copy().sort_values('t_utc')

    # dx, dy: displacement from previous point (first point → 0)
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)

    # COG: circular encoding
    cog_rad = np.deg2rad(df['cog'])
    df['cog_sin'] = np.sin(cog_rad)
    df['cog_cos'] = np.cos(cog_rad)

    return df


def normalize_dataset(df: pd.DataFrame, scaler=None, fit=False, verbose=True):
    """Compute features, then z-score numeric columns."""

    # Per-vessel feature computation
    df = df.groupby('vessel_id', group_keys=False).apply(compute_features, include_groups=False)

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

    # Drop replaced columns, keep everything else (vessel_id, t_utc, x, y for reference)
    df_out = pd.concat([df.drop(columns=NUMERIC_FEATURES), df_numeric], axis=1)

    if verbose:
        print(f"  Normalized {len(df):,} rows")
    return df_out, scaler


if __name__ == "__main__":
    input_dir  = Path("output/03_sampled")
    output_dir = Path("output/04_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {input_dir}")
        raise SystemExit(1)

    # TODO: fit scaler on train split only, apply to val/test
    scalers = {}
    total_rows = 0

    for src in input_files:
        print(f"\n{'='*60}")
        print(f"Normalizing: {src.name}")
        print(f"{'='*60}")

        df = pd.read_csv(src, parse_dates=['t_utc']).copy()
        n_rows = len(df)

        df_norm, scaler = normalize_dataset(df, fit=True, verbose=True)
        scalers[src.name] = {
            'mean':  scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'features': NUMERIC_FEATURES
        }

        dst = output_dir / src.name
        df_norm.to_csv(dst, index=False)
        print(f"Saved: {dst}")

        total_rows += n_rows

    joblib.dump(scalers, output_dir / "scalers.pkl")

    print(f"\n{'='*60}")
    print(f"NORMALIZED {total_rows:,} rows across {len(input_files)} files")
    print(f"Scalers saved to output/04_normalized/")
    print(f"{'='*60}")
