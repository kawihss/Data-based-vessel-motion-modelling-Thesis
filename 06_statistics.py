import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

#todo: add context distribution per station

# Create output folder if it doesn't exist
if not os.path.exists("output/06_statistics"):
    os.makedirs("output/06_statistics")

# Count raw AIS messages 
print("Counting raw AIS messages...")
total_raw = 0
for file in glob.glob("output/01_raw/*.csv"):
    # We only read one column to save time/memory, but chunk it to avoid memory overload
    for chunk in pd.read_csv(file, usecols=[0], chunksize=1000000):
        total_raw += len(chunk)

print(f"Total raw messages: {total_raw}")
print("-" * 50)

# Process data
print("Processing final trajectories...")

kiel_vessels, brem_vessels, wed_vessels = set(), set(), set()
kiel_tracks, brem_tracks, wed_tracks = set(), set(), set()
kiel_pts, brem_pts, wed_pts = 0, 0, 0

plot_data = []
target_cols = ['vessel_id', 'track_id', 'context', 'sog', 'cog', 'rot']

files = glob.glob("output/05_normalized/*.csv")

for file in files: 
    filename = file.lower()
    if 'kiel' in filename:
        station_name = 'Kiel'
    elif 'bremerhaven' in filename:
        station_name = 'Bremerhaven'
    else:
        station_name = 'Wedel'

    # Check available columns
    actual_cols = pd.read_csv(file, nrows=0).columns.tolist()
    valid_cols = [c for c in target_cols if c in actual_cols]

    for chunk_idx, df in enumerate(pd.read_csv(file, usecols=valid_cols, chunksize=100000)):
        if df.empty:
            continue

        vessels = df['vessel_id'].unique() if 'vessel_id' in df.columns else []
        tracks = df['track_id'].unique() if 'track_id' in df.columns else []

        if station_name == 'Kiel':
            kiel_vessels.update(vessels)
            kiel_tracks.update(tracks)
            kiel_pts += len(df)
        elif station_name == 'Bremerhaven':
            brem_vessels.update(vessels)
            brem_tracks.update(tracks)
            brem_pts += len(df)
        else:
            wed_vessels.update(vessels)
            wed_tracks.update(tracks)
            wed_pts += len(df)

        # we take a small sample per chunk.
        # And we stop collecting when we have enough global points for the plot (e.g., 100k)
        if len(plot_data) < 1000: # Max 1000 Chunks of ~100 rows = ~100,000 points
            df_sample = df.sample(n=min(100, len(df)), random_state=42).copy()
            df_sample['station'] = station_name
            plot_data.append(df_sample)


print(f"Kiel        | Vessels: {len(kiel_vessels)} | Tracks: {len(kiel_tracks)} ")
print(f"Bremerhaven | Vessels: {len(brem_vessels)} | Tracks: {len(brem_tracks)} ")
print(f"Wedel       | Vessels: {len(wed_vessels)} | Tracks: {len(wed_tracks)} ")
print("-" * 50)

df_plot = pd.concat(plot_data, ignore_index=True)

#  Create Histograms
stations = ['Kiel', 'Bremerhaven', 'Wedel']

fig, axes = plt.subplots(nrows=4, ncols=len(stations), figsize=(18, 16))

for i, station in enumerate(stations):
    station_data = df_plot[df_plot['station'] == station]
    
    if station_data.empty:
        continue

    if 'sog' in station_data.columns:
        axes[0, i].hist(station_data['sog'].dropna(), bins=40, color='blue', alpha=0.6)
    axes[0, i].set_title(f"{station} - SOG (knots)")
    
    if 'cog' in station_data.columns:
        axes[1, i].hist(station_data['cog'].dropna(), bins=40, color='orange', alpha=0.6)
    axes[1, i].set_title(f"{station} - COG (degrees)")

    if 'rot' in station_data.columns:
        axes[2, i].hist(station_data['rot'].dropna(), bins=40, color='green', alpha=0.6)
    axes[2, i].set_title(f"{station} - ROT (deg/min)")

    if 'context' in station_data.columns:
        context_counts = station_data['context'].value_counts()
        context_counts.plot(kind='bar', ax=axes[3, i], color='purple', alpha=0.6)
    axes[3, i].set_title(f"{station} - Context Distribution")
    axes[3, i].set_xlabel("Count (Sampled)")

plt.tight_layout()
plt.savefig("output/06_statistics/simple_histograms.pdf")
print("Saved histograms to output/06_statistics/simple_histograms.pdf", flush=True)