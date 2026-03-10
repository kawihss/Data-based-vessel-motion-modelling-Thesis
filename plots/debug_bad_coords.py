# Debug script — find lat/lon values causing UTM inf

import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path

def find_bad_coords(csv_path):
    print(f"Analyzing {csv_path}")
    df = pd.read_csv(csv_path)

    # try projection
    median_lon = df['lon'].median()
    zone = int((median_lon + 180) / 6) + 1
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{zone}", always_xy=True)

    # test projection on every row
    x, y = transformer.transform(df['lon'], df['lat'])

    # find rows where result is inf
    bad_mask = ~np.isfinite(x) | ~np.isfinite(y)
    bad_rows = df[bad_mask][['lat', 'lon']]

    print(f"Bad projection rows: {len(bad_rows)}")
    if len(bad_rows) > 0:
        print(bad_rows.head(20))
        print(f"\nlat range: {df['lat'].min():.3f} to {df['lat'].max():.3f}")
        print(f"lon range: {df['lon'].min():.3f} to {df['lon'].max():.3f}")

if __name__ == "__main__":
    find_bad_coords("output/01_raw/processed_ais_bremerhaven_20180404.csv")
