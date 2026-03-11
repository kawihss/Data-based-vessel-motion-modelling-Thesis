# 03_sample.py — time resampling (10s German, 60s NOAA)
# Input:  output/02_cleaned/*.csv (with x,y but without lat,lon)
# Output: output/03_sampled/*.csv (resampled trajectories)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Global resampling frequencies (seconds)
GERMAN_FREQ_S = 10
NOAA_FREQ_S   = 60


def infer_freq_from_name(name: str) -> int:
    """Map filename to resampling frequency in seconds."""
    lower = name.lower()
    if 'kiel' in lower or 'bremerhaven' in lower:
        return GERMAN_FREQ_S
    if 'marinecadastre' in lower or 'mississippi' in lower:
        return NOAA_FREQ_S
    # default: be conservative
    return GERMAN_FREQ_S

def split_long_gaps(df: pd.DataFrame, max_gap_s: int) -> pd.DataFrame:
    """Split vessel trajectories into segments at large time gaps."""
    df = df.sort_values(['vessel_id', 't_utc']).copy()
    df['dt'] = df.groupby('vessel_id')['t_utc'].diff().dt.total_seconds()
    
    # mark segment breaks
    df['gap_break'] = (df['dt'] > max_gap_s) | df['dt'].isna()
    df['segment_id'] = df.groupby('vessel_id')['gap_break'].cumsum()
    df['vessel_id'] = (df['vessel_id'].astype(str) + '_' + 
                       df['segment_id'].astype(str))
    
    return df.drop(columns=['dt', 'gap_break', 'segment_id'])

def resample_dataset(df: pd.DataFrame, freq_s: int) -> pd.DataFrame:
    """Resample all vessel trajectories in df to fixed freq_s using time-linear interp."""
    df = split_long_gaps(df, max_gap_s=3 * freq_s)
    df = df.sort_values(['vessel_id', 't_utc']).copy()

    resampled_parts = []

    for vid, g in df.groupby('vessel_id'):
        g = g.sort_values('t_utc')
        real_times = g['t_utc'].values  # ++ save original timestamps

        g = g.set_index('t_utc')

        t_start = g.index[0].ceil(f'{freq_s}s')
        t_end   = g.index[-1].floor(f'{freq_s}s')
        if t_end <= t_start:
            continue
        new_idx = pd.date_range(t_start, t_end, freq=f'{freq_s}s')

        g_resampled = g.reindex(g.index.union(new_idx))
        num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
        if 'dt' in num_cols:
            num_cols.remove('dt')
        spatial_cols     = [c for c in ['x', 'y'] if c in num_cols]
        non_spatial_cols = [c for c in num_cols if c not in spatial_cols]

        if spatial_cols:
            method = 'spline' if len(g) >= 4 else 'time'
            g_resampled[spatial_cols] = g_resampled[spatial_cols].interpolate(#we might get a warning for sparse marinecadastra
            #waring when points are coliniear. can be ignored, fallback is time interpolation which will work but be less smooth.
                method= method, order=3
            )
        if non_spatial_cols:
            g_resampled[non_spatial_cols] = g_resampled[non_spatial_cols].interpolate(
                method='time'
            )        
        g_resampled = g_resampled.loc[new_idx]

        # ++ compute dt_quality: seconds to nearest real AIS point
        new_times_s   = new_idx.astype(np.int64) / 1e9
        real_times_s  = pd.DatetimeIndex(real_times).astype(np.int64) / 1e9
        idx           = np.searchsorted(real_times_s, new_times_s).clip(0, len(real_times_s) - 1)
        idx_prev      = (idx - 1).clip(0, len(real_times_s) - 1)
        dist_next     = np.abs(new_times_s - real_times_s[idx])
        dist_prev     = np.abs(new_times_s - real_times_s[idx_prev])
        g_resampled['dt'] = np.minimum(dist_next, dist_prev)  #

        g_resampled['vessel_id'] = vid
        g_resampled['dataset']   = g['dataset'].iloc[0]
        resampled_parts.append(g_resampled.reset_index().rename(columns={'index': 't_utc'}))

    if not resampled_parts:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(resampled_parts, ignore_index=True)

    # recompute dt on the resampled grid
    out = out.sort_values(['vessel_id', 't_utc'])
    out['dt'] = out['dt'].fillna(0)

    return out


if __name__ == '__main__':
    input_dir  = Path('output/02_cleaned')
    output_dir = Path('output/03_sampled')
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob('*.csv'))
    if not input_files:
        print(f'No CSV files found in {input_dir}')
        raise SystemExit(1)

    total_in = total_out = 0

    for src in input_files:
        print(f"\n{'='*60}")
        print(f'Resampling: {src.name}')
        print(f"{'='*60}")

        df = pd.read_csv(src, parse_dates=['t_utc']).copy()
        n_in = len(df)

        freq_s = infer_freq_from_name(src.name)
        print(f'  Using frequency: {freq_s}s')

        df_resampled = resample_dataset(df, freq_s=freq_s)
        n_out = len(df_resampled)

        dst = output_dir / src.name
        df_resampled.to_csv(dst, index=False)
        print(f'  Saved: {dst}  ({n_out:,} rows from {n_in:,})')

        total_in  += n_in
        total_out += n_out

    print(f"\n{'='*60}")
    print(f'DONE  raw rows: {total_in:,}  →  resampled rows: {total_out:,}')
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")
