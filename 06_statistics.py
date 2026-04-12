import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # prevents segfault on VERA
#chunking because we had some segfaults when trying to read all the data at once
#uses only a sample for the plots, but counts all the trajectories for the exact numbers in the bar plot 


#Counting raw AIS messages...
#Total raw messages: 154269746
#--------------------------------------------------
#Processing final trajectories...
#Kiel        | Vessels: 41201 | Tracks: 3266235 
#Bremerhaven | Vessels: 27526 | Tracks: 2162097 
#Wedel       | Vessels: 14390 | Tracks: 1385134
#--------------------------------------------------
#Distinct Trajectories per Context:
#--- Kiel ---
#harbour    2725942
#channel     476853
#lock         62404
#river         1036
#--- Bremerhaven ---
#harbour    2028134
#river        77747
#lock         56199
#channel         17
#--- Wedel ---
#river    1385134

# Create output folder if it doesn't exist
os.makedirs("output/06_statistics", exist_ok=True)

print("Counting raw AIS messages...")
total_raw = sum(len(chunk) for file in glob.glob("output/01_raw/*.csv") 
                for chunk in pd.read_csv(file, usecols=[0], chunksize=1000000))

print(f"Total raw messages: {total_raw}")

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

        # Grab a random sample for the plots, rn set to 100%
        df_sample = df.sample(frac=1, random_state=42).copy()
        df_sample['station'] = station_name
        plot_data.append(df_sample)


print(f"Kiel        | Vessels: {len(kiel_vessels)} | Tracks: {len(kiel_tracks)} ")
print(f"Bremerhaven | Vessels: {len(brem_vessels)} | Tracks: {len(brem_tracks)} ")
print(f"Wedel       | Vessels: {len(wed_vessels)} | Tracks: {len(wed_tracks)} ")

# sets of track_ids into final counts
final_context_counts = {
    station: pd.Series({ctx: len(tracks) for ctx, tracks in ctx_dict.items()})
    for station, ctx_dict in true_contexts.items()
}

print("Distinct Trajectories per Context:")
for station, counts in final_context_counts.items():
    print(f"--- {station} ---")
    print(counts.sort_values(ascending=False).to_string() if not counts.empty else "No context data")

df_plot = pd.concat(plot_data, ignore_index=True)

# Histograms
stations = ['Kiel', 'Bremerhaven', 'Wedel']
fig, axes = plt.subplots(nrows=4, ncols=len(stations), figsize=(18, 16))

for i, station in enumerate(stations):
    station_data = df_plot[df_plot['station'] == station]
    
    # sampled distributions for SOG, COG, ROT
    axes[0, i].hist(station_data['sog'].dropna(), bins=40, color='blue', alpha=0.6, density=True)
    axes[0, i].set_title(f"{station} - SOG (knots)")
    
    axes[1, i].hist(station_data['cog'].dropna(), bins=40, color='orange', alpha=0.6, density=True)
    axes[1, i].set_title(f"{station} - COG (degrees)")

    axes[2, i].hist(station_data['rot'].dropna(), bins=40, color='green', alpha=0.6, density=True)
    axes[2, i].set_title(f"{station} - ROT (deg/min)")

    # distinct trajectory counts for context
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

plt.close(fig) 


# COG Polar Projection Heatmap
print("Generating COG polar projections...")
fig_polar, axes_polar = plt.subplots(1, len(stations), figsize=(18, 6), subplot_kw={'projection': 'polar'}, constrained_layout=True)

for i, station in enumerate(stations):
    station_cog_data = df_plot[df_plot['station'] == station]['cog'].dropna().values
    
    if len(station_cog_data) > 0:
        cog_rad = np.deg2rad(station_cog_data)
        
        # Create a polar histogram 
        n, bins, patches = axes_polar[i].hist(cog_rad, bins=36, alpha=0.8, density=True)
        
        axes_polar[i].set_theta_zero_location('N')
        axes_polar[i].set_theta_direction(-1)
        
        #  heatmap coloring based on bin frequency
        if n.max() > 0:
            fracs = n / n.max()
            for frac, patch in zip(fracs, patches):
                color = plt.cm.YlOrRd(frac) # Yellow-Orange-Red colormap
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)
            
    axes_polar[i].set_title(f"{station} - COG Polar Heatmap", pad=20, fontsize=14)

plt.tight_layout() # formating issues with label on top, ignore warning, it looks good now
polar_output_path = "output/06_statistics/cog_polar_heatmap.pdf"
plt.savefig(polar_output_path, bbox_inches='tight')
plt.close(fig_polar) # Clean up memory again
print(f"Saved COG polar heatmap to {polar_output_path}", flush=True)
