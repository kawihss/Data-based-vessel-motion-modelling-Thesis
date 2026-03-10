# 04_normalize.py — StandardScaler normalization (placeholder)
# Input:  output/03_sampled/*.csv
# Output: output/04_normalized/*.csv + scaler.pkl

#PALCEHOLDER using z score normalization DO RESEARCH ON BEST PRACTICES FOR NORMALIZATION IN TIME SERIES TRAJECTORY 


import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

NUMERIC_FEATURES = ['x', 'y', 'sog', 'cog', 'dt']  # to be normalized

def normalize_dataset(df: pd.DataFrame, scaler=None, fit=False, verbose=True):
    """Normalize numeric features. fit=True computes scaler params on this data."""
    if fit:
        scaler = StandardScaler()
        scaler.fit(df[NUMERIC_FEATURES])
        if verbose:
            print("  Scaler fitted:")
            for feat, mu, sigma in zip(NUMERIC_FEATURES, scaler.mean_, scaler.scale_):
                print(f"    {feat}: μ={mu:.1f}, σ={sigma:.1f}")

    # Apply normalization
    df_numeric = pd.DataFrame(
        scaler.transform(df[NUMERIC_FEATURES]),
        columns=[f'{feat}_norm' for feat in NUMERIC_FEATURES],
        index=df.index
    )

    # Combine with non-numeric columns
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

    # Placeholder: normalize each file independently
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
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }

        dst = output_dir / src.name
        df_norm.to_csv(dst, index=False)
        print(f"✓ Saved: {dst}")

        total_rows += n_rows

   
    joblib.dump(scalers, output_dir / "scalers.pkl")

    print(f"\n{'='*60}")
    print(f"NORMALIZED {total_rows:,} rows across {len(input_files)} files")
    print(f"Scalers saved to output/04_normalized/")
    print(f"{'='*60}")
