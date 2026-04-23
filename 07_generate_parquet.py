import time
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("output/05_normalized")
OUTPUT_DIR = Path("output/07_parquet")
FILE_PATTERN = "*.csv"
COMPRESSION = "snappy"
OVERWRITE = True

FLOAT32_COLS = [
    'x', 'y', 'sog', 'cog', 'rot', 'dt',
    'dx', 'dy', 'cog_sin', 'cog_cos',
    'dx_norm', 'dy_norm', 'sog_norm', 'cog_sin_norm', 'cog_cos_norm', 'dt_norm', 'rot_norm',
]
STRING_COLS = ['t_utc', 'role', 'vessel_id', 'context']


def _normalize_export_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if 'track_id' in df.columns:
        df['track_id'] = pd.to_numeric(df['track_id'], errors='raise').astype('int64')

    for col in FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype('string')

    return df


def convert_csv_to_parquet(input_dir: Path, output_dir: Path):
    csv_files = sorted(input_dir.glob(FILE_PATTERN))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_csv_bytes = 0
    total_parquet_bytes = 0

    start_time = time.perf_counter()
    print(f"Converting {len(csv_files)} file(s) from {input_dir} to {output_dir}")

    for idx, src in enumerate(csv_files, start=1):
        dst = output_dir / f"{src.stem}.parquet"
        if dst.exists() and not OVERWRITE:
            print(f"[{idx}/{len(csv_files)}] Skipping existing {dst.name}")
            continue

        t0 = time.perf_counter()
        df = pd.read_csv(src, low_memory=False)
        df = _normalize_export_dtypes(df)
        df.to_parquet(dst, index=False, compression=COMPRESSION)
        dt = time.perf_counter() - t0

        n_rows = len(df)
        csv_mb = src.stat().st_size / (1024 * 1024)
        parquet_mb = dst.stat().st_size / (1024 * 1024)

        total_rows += n_rows
        total_csv_bytes += src.stat().st_size
        total_parquet_bytes += dst.stat().st_size

        print(
            f"[{idx}/{len(csv_files)}] {src.name} -> {dst.name} | "
            f"rows={n_rows:,} | {csv_mb:.2f}MB -> {parquet_mb:.2f}MB | {dt:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    if total_csv_bytes > 0:
        ratio = total_parquet_bytes / total_csv_bytes
    else:
        ratio = float("nan")

    print("Parquet export complete")
    print(f"  Files converted : {len(csv_files)}")
    print(f"  Total rows      : {total_rows:,}")
    print(f"  Total CSV size  : {total_csv_bytes / (1024 * 1024):.2f}MB")
    print(f"  Total Parquet   : {total_parquet_bytes / (1024 * 1024):.2f}MB")
    print(f"  Size ratio      : {ratio:.3f}")
    print(f"  Total runtime   : {total_time:.2f}s")


if __name__ == "__main__":
    convert_csv_to_parquet(INPUT_DIR, OUTPUT_DIR)
