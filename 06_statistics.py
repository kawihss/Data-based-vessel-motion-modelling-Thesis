import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from collections import defaultdict

#chunking because we had some segfaults when trying to read all the data at once
#uses only a sample for the plots, but counts all the trajectories for the exact numbers in the bar plot 

# Create output folder if it doesn't exist
os.makedirs("output/06_statistics", exist_ok=True)

# Count raw AIS messages 
print("Counting raw AIS messages...")
total_raw = sum(len(chunk) for file in glob.glob("output/01_raw/*.csv") 
                for chunk in pd.read_csv(file, usecols=[0], chunksize=1000000))

print(f"Total raw messages: {total_raw}")
print("-" * 50)

# Process data
print("Processing final trajectories...")

kiel_vessels, brem_vessels, wed_vessels = set(), set(), set()
kiel_tracks, brem_tracks, wed_tracks = set(), set(), set()
kiel_pts, brem_pts, wed_pts = 0, 0, 0

# Dictionary of sets to store unique track_ids per context, per station
true_contexts = {
    'Kiel': defaultdict(set), 
    'Bremerhaven': defaultdict(set), 
    'Wedel': defaultdict(set)
}

plot_data = []
target_cols = ['vessel_id', 'track_id', 'context', 'sog', 'cog', 'rot']

for file in glob.glob("output/05_normalized/*.csv"): 
    filename = file.lower()
    if 'kiel' in filename:
        station_name = 'Kiel'
    elif 'bremerhaven' in filename:
        station_name = 'Bremerhaven'
    else:
        station_name = 'Wedel'

    for chunk_idx, df in enumerate(pd.read_csv(file, usecols=target_cols, chunksize=100000)):
        
        vessels = df['vessel_id'].unique()
        tracks = df['track_id'].unique()

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

        for ctx, unique_tracks in df.groupby('context')['track_id'].unique().items():
            true_contexts[station_name][ctx].update(unique_tracks)

        # Grab a random 5% (0.05) of EVERY chunk
        df_sample = df.sample(frac=0.05, random_state=42).copy()
        df_sample['station'] = station_name
        plot_data.append(df_sample)


print(f"Kiel        | Vessels: {len(kiel_vessels)} | Tracks: {len(kiel_tracks)} ")
print(f"Bremerhaven | Vessels: {len(brem_vessels)} | Tracks: {len(brem_tracks)} ")
print(f"Wedel       | Vessels: {len(wed_vessels)} | Tracks: {len(wed_tracks)} ")
print("-" * 50)

# Convert sets of track_ids into final counts
final_context_counts = {
    station: pd.Series({ctx: len(tracks) for ctx, tracks in ctx_dict.items()})
    for station, ctx_dict in true_contexts.items()
}

# Print exact trajectory counts to console
print("Distinct Trajectories per Context:")
for station, counts in final_context_counts.items():
    print(f"--- {station} ---")
    print(counts.sort_values(ascending=False).to_string() if not counts.empty else "No context data")
print("-" * 50)

# Combine sampled data for continuous distributions
df_plot = pd.concat(plot_data, ignore_index=True)

# Create Histograms
stations = ['Kiel', 'Bremerhaven', 'Wedel']
fig, axes = plt.subplots(nrows=4, ncols=len(stations), figsize=(18, 16))

for i, station in enumerate(stations):
    station_data = df_plot[df_plot['station'] == station]
    
    # Plotting sampled distributions for SOG, COG, ROT
    axes[0, i].hist(station_data['sog'].dropna(), bins=40, color='blue', alpha=0.6, density=True)
    axes[0, i].set_title(f"{station} - SOG (knots)")
    
    axes[1, i].hist(station_data['cog'].dropna(), bins=40, color='orange', alpha=0.6, density=True)
    axes[1, i].set_title(f"{station} - COG (degrees)")

    axes[2, i].hist(station_data['rot'].dropna(), bins=40, color='green', alpha=0.6, density=True)
    axes[2, i].set_title(f"{station} - ROT (deg/min)")

    # Plot the distinct trajectory counts for context
    counts_series = final_context_counts.get(station, pd.Series(dtype=int))
    
    if not counts_series.empty:
        counts_sorted = counts_series.sort_values(ascending=False)
        ax_bar = counts_sorted.plot(kind='bar', ax=axes[3, i], color='purple', alpha=0.6)
        axes[3, i].set_title(f"{station} - Unique Trajectories per Context")
        axes[3, i].set_xlabel("Distinct Trajectories (Unsampled)")
        
        for container in ax_bar.containers:
            ax_bar.bar_label(container, fmt='%.0f')

plt.tight_layout()
plt.savefig("output/06_statistics/simple_histograms.pdf")
print("Saved histograms to output/06_statistics/simple_histograms.pdf", flush=True)