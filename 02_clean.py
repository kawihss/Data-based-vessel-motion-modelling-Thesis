# 02_clean.py — AIS Cleaning: coordinate conversion, deduplication, MMSI hashing
# Input:  output/01_raw/*.csv
# Output: output/02_cleaned/*.csv this is what we can publish, features output in 04_normalize.py

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime
from pyproj import Transformer


# ── Hashing ──────────────────────────────────────────────────────────────────
def hash_mmsi(mmsi) -> str:
    """One-way SHA-256 hash of MMSI → 12-char anonymous vessel ID."""
    return hashlib.sha256(str(int(mmsi)).encode()).hexdigest()[:12]


# ── UTM conversion ───────────────────────────────────────────────────────────
def add_utm_coordinates(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    df = df.copy()
    median_lon = df['lon'].median()
    median_lat = df['lat'].median()
    zone = int((median_lon + 180) / 6) + 1
    epsg = f"326{zone:02d}" if median_lat >= 0 else f"327{zone:02d}"

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    valid = df['lat'].notna() & df['lon'].notna()
    x = np.full(len(df), np.nan)
    y = np.full(len(df), np.nan)
    x[valid], y[valid] = transformer.transform(
        df.loc[valid, 'lon'].values,
        df.loc[valid, 'lat'].values
    )
    df['x'] = x
    df['y'] = y

    hemi = 'N' if median_lat >= 0 else 'S'
    if verbose:
        print(f"  UTM Zone {zone}{hemi} (EPSG:{epsg})")
        print(f"  x: {df['x'].min():.0f} – {df['x'].max():.0f} m")
        print(f"  y: {df['y'].min():.0f} – {df['y'].max():.0f} m")
    return df


def remove_position_jumps(df: pd.DataFrame, threshold=2.0) -> pd.DataFrame:
    """Delete rows where displacement exceeds threshold × speed-implied distance."""
    df = df.copy().sort_values('t_utc').reset_index(drop=True)

    t = df['t_utc'].astype(np.int64) / 1e9          # → seconds
    dt = np.diff(t, prepend=np.nan)
    dx = np.diff(df['x'].values, prepend=np.nan)
    dy = np.diff(df['y'].values, prepend=np.nan)
    actual_dist = np.sqrt(dx**2 + dy**2)             # metres

    speed_ms = df['sog'].values * 1852 / 3600        # knots → m/s
    speed_avg = (speed_ms + np.roll(speed_ms, 1)) / 2
    speed_avg[0] = speed_ms[0]
    expected_dist = speed_avg * np.abs(dt)

    is_jump = actual_dist > threshold * np.maximum(expected_dist, 10)
    is_jump[0] = False

    return df[~is_jump].reset_index(drop=True)


def remove_fast_vessels(df: pd.DataFrame, max_sog=30, verbose=True) -> pd.DataFrame:
    """Remove all rows belonging to vessels that ever exceed max_sog knots."""
    fast_ids = df.groupby('vessel_id')['sog'].max()
    fast_ids = fast_ids[fast_ids > max_sog].index
    mask = df['vessel_id'].isin(fast_ids)
    if verbose:
        print(f"  Fast vessel filter: removed {mask.sum():,} rows "
              f"({fast_ids.nunique()} vessels exceeding {max_sog} kn)")
    return df[~mask].reset_index(drop=True)

# ── Main clean function ───────────────────────────────────────────────────────
def clean(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    n0 = len(df)

    # 1. Geographic bounds filter (dataset-specific). very vague for now, just to catch obvious outliers and coordinate errors. to be reifined later for missisipi dataset.
    n_pre_geo = len(df)  

    if 'bremerhaven' in df['dataset'].iloc[0].lower():
        mask = df['lat'].between(50, 60) & df['lon'].between(5, 10)
    elif 'kiel' in df['dataset'].iloc[0].lower():
        mask = df['lat'].between(50, 62) & df['lon'].between(8, 12)
    elif 'marinecadastre' in df['dataset'].iloc[0].lower():
        mask = df['lat'].between(28.8, 35.2) & df['lon'].between(-91.0, -88.8) # polygon later
    else:
        mask = pd.Series(True, index=df.index)

    df = df[mask].copy()
    if verbose:
        print(f"  Geographic filter kept {len(df):,} / {n_pre_geo:,} rows")


    # 2. Anonymize MMSI → hashed vessel_id
    df['vessel_id'] = df['vessel_id'].apply(hash_mmsi)
    if verbose:
        print(f"  Hashed {df['vessel_id'].nunique()} unique vessel IDs")

    
    # 3. Remove duplicates — keep row with most non-null values
    n_before_dedup = len(df)

    df['_non_null'] = df.notna().sum(axis=1)
    df = df.sort_values('_non_null', ascending=False)
    df = df.drop_duplicates(subset=['vessel_id', 't_utc'], keep='first')
    df = df.drop(columns='_non_null')

    if verbose:
        print(f"  Duplicates removed: {n_before_dedup - len(df):,}")

    # 4. UTM coordinate conversion
    df = add_utm_coordinates(df, verbose=verbose)

    # Reorder columns: insert x, y after lon
    cols = list(df.columns)
    if 'x' in cols and 'y' in cols:
        cols = [c for c in cols if c not in ('x', 'y')]
        lon_idx = cols.index('lon')
        cols = cols[:lon_idx+1] + ['x', 'y'] + cols[lon_idx+1:]
        df = df[cols]

    #Remove position jumps per vessel ──────────────────────────────
    n_before_jumps = len(df)
    df = df.groupby('vessel_id', group_keys=False).apply(remove_position_jumps)
    if verbose:
        print(f"  Jump filter removed: {n_before_jumps - len(df):,}")

    df = remove_fast_vessels(df, max_sog=30, verbose=verbose)

    if verbose:
        print(f"  Rows: {n0:,} → {len(df):,} (removed {n0 - len(df):,})")




    df = df.drop(columns=['lat', 'lon'])

    return df.sort_values(['vessel_id', 't_utc']).reset_index(drop=True)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_dir  = Path("output/01_raw")
    output_dir = Path("output/02_cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {input_dir}")
        exit(1)

    total_in = total_out = 0

    for src in input_files:
        print(f"\n{'='*60}")
        print(f"Cleaning: {src.name}")
        print(f"{'='*60}")

        df = pd.read_csv(src, parse_dates=['t_utc']).copy()
        n_in = len(df)

        df = clean(df, verbose=True)
        n_out = len(df)

        dst = output_dir / src.name
        df.to_csv(dst, index=False)
        print(f"✓ Saved: {dst}")

        total_in  += n_in
        total_out += n_out

    
    print(f"\n{'='*60}")
    print(f"DONE  {total_in:,} → {total_out:,} rows")
    print(f"{'='*60}")

