#Not up to date

# AIS Vessel Trajectory Preprocessing Pipeline

A five-step preprocessing pipeline for raw AIS (Automatic Identification System) data,
producing clean, resampled, and normalised trajectory samples for deep learning-based
vessel motion prediction.

Part of the bachelor's thesis:
**"Deep Learning-based Motion Models for Multi-Target Tracking of Inland Vessels using AIS Data"**
Dustin Klein, RWTH Aachen University, Chair of AI Methodology (Informatik 14) and institute of Automatic Control

---

## Overview

Raw AIS logs from German coastal stations (Kiel, Bremerhaven) and the U.S. NOAA Marine
Cadastre (Upper Mississippi River) are transformed into labelled context/prediction
trajectory pairs suitable for sequence models (e.g. LSTM, Transformer).

```

Data/
├── Kiel/              ← German BSH semicolon-delimited AIS logs
├── Bremerhaven/       ← German BSH semicolon-delimited AIS logs
└── Mississippi/       ← NOAA Marine Cadastre CSV

output/
├── 01_raw/            ← Extracted unified CSVs
├── 02_cleaned/        ← Filtered, deduplicated, UTM-projected
├── 03_sampled/        ← Uniformly resampled (10 s / 60 s)
├── 04_trajectories/   ← Sliding-window segments, train/val/test split
└── 05_normalized/     ← Z-score normalised feature vectors + scalers.pkl

```

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_extract.py` | Parse raw AIS files to unified CSV (position types 1,2,3,18,27 only) |
| 2 | `02_clean.py` | Geo-filter, MMSI hashing, dedup, UTM projection, kinematic outlier removal |
| 3 | `03_sample.py` | Resample to uniform 30s grid via spline + linear interpolation |
| 4 | `04_track_segmentation.py` | Sliding window (10 min ctx + 2 min pred), speed filters, vessel-level split |
| 5 | `05_normalize.py` | Compute dx/dy, COG sin/cos, Δt; fit StandardScaler on train only |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Download raw data

```
python 00_download.py
```


### 3. Run the full pipeline

```bash
python 00_full_preprocessing_pipeline.py
```

Or run individual steps:

```bash
python 01_extract.py
python 02_clean.py
python 03_sample.py
python 04_track_segmentation.py
python 05_normalize.py
```


---

## Output Format

### After Step 5 (`output/05_normalized/`)

Each CSV contains one row per AIS point, labelled as `context` or `prediction`:


| Column | Description |
| :-- | :-- |
| `track_id` | Unique sliding-window segment ID |
| `role` | `context` (10 min input) or `prediction` (2 min target) |
| `t_utc` | UTC timestamp |
| `x`, `y` | UTM coordinates (metres) |
| `sog`, `cog` | Raw speed (kn) and course (°) |
| `dx_norm`, `dy_norm` | Normalised relative displacements |
| `sog_norm` | Normalised speed over ground |
| `cog_sin_norm`, `cog_cos_norm` | Normalised circular COG encoding |
| `dt_norm` | Normalised temporal distance to nearest raw observation |

Files are split by dataset and partition: `train_*.csv`, `val_*.csv`, `test_*.csv`.

The fitted scaler is saved as `output/05_normalized/scalers.pkl`.

---

## Configuration

Key parameters are defined at the top of each script:

**`03_sample.py`**

```python
GERMAN_FREQ_S = 00   # resampling interval
```

**`04_track_segmentation.py`**

```python
WINDOW_DUR        = timedelta(minutes=10)  # context window length
PRED_HORIZON      = timedelta(minutes=2)   # prediction horizon
STRIDE_DUR        = timedelta(minutes=2)   # sliding window stride
MIN_SOG           = 1.0   # knots — anchoring detection threshold
LOW_SPEED_THRESH  = 2.0   # knots — low-speed threshold
LOW_SPEED_FRAC_MAX = 0.3  # max fraction of low-speed points per track
```


---

## Dataset Split Strategy

Splits are performed at **vessel level** to prevent data leakage from overlapping
sliding-window segments:

- **Train** — 60 % of unique vessels
- **Validation** — 20 % of unique vessels
- **Test** — 20 % of unique vessels

Vessel IDs are shuffled with `random_seed=42` before splitting.

---

## Privacy

MMSI numbers are irreversibly anonymised using a one-way **SHA-256 hash**
(truncated to 12 hex characters) during the cleaning step. The cleaned output
in `output/02_cleaned/` onwards contains no raw MMSI values and is safe
for publication.

---

## Project Structure

```
.
├── 00_full_preprocessing_pipeline.py  ← Run all steps in sequence
├── 01_extract.py
├── 02_clean.py
├── 03_sample.py
├── 04_track_segmentation.py
├── 05_normalize.py
├── Data/                              ← Raw AIS input (not committed)
└── output/                            ← Generated artefacts (not committed)
```


---

## Citation

If you use this pipeline, please cite:

TODO

---

## License

TODO

