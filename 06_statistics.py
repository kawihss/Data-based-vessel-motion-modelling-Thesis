import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # prevents segfault on VERA
#chunking because we had some segfaults when trying to read all the data at once
#uses only a sample for the plots, but counts all the trajectories for the exact numbers in the bar plot 

#factor 1.1
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

#with factor 1: acceptable 
#Counting raw AIS messages...
#Total raw messages: 154269746
#Processing final trajectories...
##Kiel        | Vessels: 40249 | Tracks: 2919666 
#Bremerhaven | Vessels: 26908 | Tracks: 1955845 
#Wedel       | Vessels: 14109 | Tracks: 1332553 
#Distinct Trajectories per Context:
#--- Kiel ---
#harbour    2397340
#channel     470044
#lock         51534
#river          748
#--- Bremerhaven ---
#harbour    1863012
#river        70787
#lock         22023
#channel         23
#--- Wedel ---
#river    1332553



# Create output folder if it doesn't exist
os.makedirs("output/06_statistics", exist_ok=True)

# total_raw = sum(len(chunk) for file in glob.glob("output/01_raw/*.csv") 
#                 for chunk in pd.read_csv(file, usecols=[0], chunksize=1000000))
total_raw = 153681423
print(f"Total raw messages: {total_raw} (hardcoded)")

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

# Violin plots for SOG and ROT per station + bar chart for trajectory counts
df_station_violin = df_plot[['station', 'sog', 'rot']].copy()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8.8))

sns.violinplot(data=df_station_violin, x='station', y='sog', order=stations,
               ax=axes[0], palette='Set2', inner='box')
axes[0].set_title("SOG (knots) per Station")
axes[0].set_xlabel("Station")
axes[0].set_ylabel("SOG (knots)")

sns.violinplot(data=df_station_violin, x='station', y='rot', order=stations,
               ax=axes[1], palette='Set2', inner='box')
axes[1].set_title("ROT (deg/min) per Station")
axes[1].set_xlabel("Station")
axes[1].set_ylabel("ROT (deg/min)")

# distinct trajectory counts for all stations combined
all_counts = pd.DataFrame([
    {'station': station, 'context': ctx, 'count': len(tracks)}
    for station, ctx_dict in true_contexts.items()
    for ctx, tracks in ctx_dict.items()
])
if not all_counts.empty:
    all_counts_pivot = all_counts.pivot(index='context', columns='station', values='count').fillna(0)
    all_counts_pivot.plot(kind='bar', ax=axes[2], color=['#66c2a5', '#fc8d62', '#8da0cb'], alpha=0.8)
    axes[2].set_title("Unique Trajectories per Context and Station")
    axes[2].set_xlabel("Context")
    axes[2].set_ylabel("Distinct Trajectories (Unsampled)")
    axes[2].tick_params(axis='x', rotation=30)
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.0f', fontsize=7)

plt.tight_layout()
plt.savefig("output/06_statistics/simple_histograms.pdf")
print("Saved station violin plots to output/06_statistics/simple_histograms.pdf", flush=True)

plt.close(fig)


# SOG and ROT per context — violin plots
df_violin = df_plot[['context', 'sog', 'rot']].dropna(subset=['context'])
context_order = sorted(df_violin['context'].unique())

fig_ctx, (ax_sog, ax_rot) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.4))

sns.violinplot(data=df_violin, x='context', y='sog', order=context_order,
               ax=ax_sog, palette='Set2', inner='box')
ax_sog.set_title("SOG (knots) per Context")
ax_sog.set_xlabel("Context")
ax_sog.set_ylabel("SOG (knots)")

sns.violinplot(data=df_violin, x='context', y='rot', order=context_order,
               ax=ax_rot, palette='Set2', inner='box')
ax_rot.set_title("ROT (deg/min) per Context")
ax_rot.set_xlabel("Context")
ax_rot.set_ylabel("ROT (deg/min)")

plt.tight_layout()
plt.savefig("output/06_statistics/context_violin.pdf")
print("Saved context violin plots to output/06_statistics/context_violin.pdf", flush=True)

plt.close(fig_ctx)


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
