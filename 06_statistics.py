import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Create output folder if it doesn't exist
if not os.path.exists("output/06_statistics"):
    os.makedirs("output/06_statistics")

# Count raw AIS messages 
print("Counting raw AIS messages...")
total_raw = 0
for file in glob.glob("output/01_raw/*.csv"):
    # We only read one column to save time/memory
    df_raw = pd.read_csv(file, usecols=[0])
    total_raw += len(df_raw)

print(f"Total raw messages: {total_raw}")
print("-" * 50)

# Process data
print("Processing final trajectories...")

# Sets for unique counts
kiel_vessels, brem_vessels, wed_vessels = set(), set(), set()
kiel_tracks, brem_tracks, wed_tracks = set(), set(), set()

# Counters for points
kiel_pts, brem_pts, wed_pts = 0, 0, 0
context_pts, prediction_pts = 0, 0

plot_data = []
cols = ['vessel_id', 'track_id', 'role', 'sog', 'cog', 'rot']

for file in glob.glob("output/05_normalized/*.csv"):
    # Load only necessary columns to keep RAM usage low
    df = pd.read_csv(file, usecols=lambda c: c in cols)
    
    if df.empty:
        continue


    # Station detection based on filename
    filename = file.lower()
    if 'kiel' in filename:
        kiel_vessels.update(df['vessel_id'].unique())
        kiel_tracks.update(df['track_id'].unique())
        kiel_pts += len(df)
        station_name = 'Kiel'
    elif 'bremerhaven' in filename:
        brem_vessels.update(df['vessel_id'].unique())
        brem_tracks.update(df['track_id'].unique())
        brem_pts += len(df)
        station_name = 'Bremerhaven'
    else:
        # Default to Wedel 
        wed_vessels.update(df['vessel_id'].unique())
        wed_tracks.update(df['track_id'].unique())
        wed_pts += len(df)
        station_name = 'Wedel'

    # Sample 5% of the data for plotting 
    df_sample = df.sample(frac=0.05, random_state=42).copy()
    df_sample['station'] = station_name
    plot_data.append(df_sample)

# Print statistics for the thesis
print("\n--- DATASET STATISTICS ---")
print(f"Kiel        | Vessels: {len(kiel_vessels)} | Tracks: {len(kiel_tracks)} ")
print(f"Bremerhaven | Vessels: {len(brem_vessels)} | Tracks: {len(brem_tracks)} ")
print(f"Wedel       | Vessels: {len(wed_vessels)} | Tracks: {len(wed_tracks)} ")
print("-" * 50)

#  Create Histograms (3 rows: SOG, COG, ROT)
print("Drawing histograms...")
df_plot = pd.concat(plot_data, ignore_index=True)
stations = ['Kiel', 'Bremerhaven', 'Wedel']

# 3 rows (SOG, COG, ROT) by N stations
fig, axes = plt.subplots(nrows=3, ncols=len(stations), figsize=(18, 12))

for i, station in enumerate(stations):
    station_data = df_plot[df_plot['station'] == station]
    
    if station_data.empty:
        continue

    axes[0, i].hist(station_data['sog'].dropna(), bins=40, color='blue', alpha=0.6)
    axes[0, i].set_title(f"{station} - SOG (knots)")
    
    axes[1, i].hist(station_data['cog'].dropna(), bins=40, color='orange', alpha=0.6)
    axes[1, i].set_title(f"{station} - COG (degrees)")

    axes[2, i].hist(station_data['rot'].dropna(), bins=40, color='green', alpha=0.6)
    axes[2, i].set_title(f"{station} - ROT (deg/min)")

plt.tight_layout()
plt.savefig("output/06_statistics/simple_histograms.pdf")
print("Saved histograms to output/06_statistics/simple_histograms.pdf")