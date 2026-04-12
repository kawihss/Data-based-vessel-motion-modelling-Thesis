# AIS Vessel Trajectory Preprocessing Pipeline

This repository contains the data preprocessing pipeline for the bachelor's thesis:
**"Short-Term Trajectory Prediction for Inland Vessels using AIS Data"**
Dustin Klein, RWTH Aachen University, Chair of AI Methodology and Institute of Automatic Control.

## Overview

The pipeline transforms raw AIS (Automatic Identification System) logs from German measurement stations (Kiel, Bremerhaven, Wedel) into normalized, context-labeled trajectory segments. These segments are formatted as input-prediction pairs suitable for sequence-to-sequence deep learning models (e.g., LSTMs). 

A major component of this pipeline is the automated assignment of geographic context (harbour, river, channel, lock) using OpenStreetMap data, allowing for context-aware model training and targeted data augmentation.

## Quick Start for Graders

The pipeline is compatible with both Windows and Linux-based operating systems. To verify the code functionality without processing the full year of 2025 data, you can run a limited execution:

1. **Install Dependencies:** Install the required Python packages by running `pip install -r requirements.txt`.
2. **Limit Data Extraction:** Open `01_extract.py` and change the limit parameter at the top to a small number (e.g., `TEST_LIMIT = 5`). This ensures the script only processes a few files per dataset.
3. **Download Sample Data:** Execute `python 00_download.py` to download the full dataset, or manually select some files by following the links you can find in the script and download them into the `Data/` directories.
4. **Run Pipeline:** Execute `python 00_full_preprocessing_pipeline.py`. This master script will clear old outputs and sequentially run steps 01 through 05, or run the full preprocessing pipeline from the starting point you specify with START_AT_STEP.


## Pipeline Architecture

The preprocessing is divided into six steps:

* **Step 0** (`00_download.py`): Fetches the raw 2025 AIS XML feeds for Kiel, Bremerhaven, and Wedel from GovData.
* **Step 1** (`01_extract.py`): Parses raw semicolon-delimited logs, filtering for dynamic position message types and valid MMSIs.
* **Step 2** (`02_clean.py`): Applies geographic bounding boxes, projects coordinates to UTM, removes kinematic outliers (e.g., position jumps), and assigns OpenStreetMap water context labels via spatial joining.
* **Step 3** (`03_sample.py`): Resamples the data to a uniform 30-second temporal grid using spline and linear interpolation, and analytically derives the Rate of Turn (ROT).
* **Step 4** (`04_track_segmentation.py`): Segments continuous tracks using a 5-minute context and 5-minute prediction sliding window. Applies a stratified train/validation/test split at the vessel level and augments minority context classes.
* **Step 5** (`05_normalize.py`): Computes relative displacements, circular COG encodings, and applies Z-score normalization using a scaler fitted exclusively on the training partition.
* **Evaluation** (`06_statistics.py`): Generates feature distributions, trajectory counts, and polar COG heatmaps used in the thesis.

## Key Configuration Parameters

The pipeline is configured to focus on maneuvering inland vessels. Key parameters include:

* **Resampling Frequency:** 30 seconds.
* **Window Dimensions:** 5-minute context window and 5-minute prediction horizon.
* **Window Stride:** 1 minute for standard data, reduced to 10 seconds for sparse lock environments.
* **Kinematic Filters:** Tracks are discarded if the maximum speed is below 0.5 knots or if more than 50% of the points are below 1.0 knot, ensuring the model learns active dynamics.

Generally, key parameters are implemented as global variables at the top of the scripts and can be easily adjusted
## Dataset Splitting and Stratification

To prevent data leakage, the dataset is split at the vessel level rather than the track segment level. Due to the high diversity in vessel behavior, a random split is insufficient.

Instead, vessels are clustered into 5 strata using K-Means based on their mean velocity, maximum ROT, and primary geographic context. The split is then performed within these strata:
* **Train:** 70%.
* **Validation:** 15%.
* **Test:** 15%.

## Data Privacy

All MMSI numbers are irreversibly anonymized during step 02 using a SHA-256 hash truncated to 12 characters. Additionally, positional privacy is maintained by expressing coordinates relative to the median position of each respective dataset, removing absolute geographic references. The data from `output/02_cleaned/` onwards is suitable for publication.