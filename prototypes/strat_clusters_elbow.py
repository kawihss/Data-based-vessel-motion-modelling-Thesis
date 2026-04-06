import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_DIR = Path('output/03_sampled')
K_RANGE = range(2, 16) # Testing from 2 to 15 clusters

def run_elbow_analysis():
    input_files = sorted(INPUT_DIR.glob('*.csv'))
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    all_inertias = []
    file_names = []

    print(f"Starting Elbow Analysis on {len(input_files)} files...")

    for src in input_files:
        print(f" Processing: {src.name}")
        df = pd.read_csv(src)
        
        if 'vessel_id' not in df.columns or 'rot' not in df.columns:
            print(f"  Skipping {src.name}: Missing required columns.")
            continue

        # 1. Vessel Profiling (Same logic as in main script)
        vessel_profiles = df.groupby('vessel_id').agg({
            'sog': 'mean',
            'rot': lambda x: x.abs().max(), # Max absolute ROT for maneuver intensity
            'context': lambda x: x.mode()[0]
        }).reset_index()

        # 2. Preprocessing
        # Convert context to numeric codes
        vessel_profiles['context_idx'] = vessel_profiles['context'].astype('category').cat.codes
        
        # Scaling features is mandatory for KMeans distance calculation
        scaler = StandardScaler()
        features = vessel_profiles[['sog', 'rot', 'context_idx']]
        scaled_features = scaler.fit_transform(features)

        # 3. Calculate Inertia for different K values
        current_file_inertia = []
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_features)
            current_file_inertia.append(km.inertia_)
        
        all_inertias.append(current_file_inertia)
        file_names.append(src.name)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # Plot individual lines for each file (gray, semi-transparent)
    for i, values in enumerate(all_inertias):
        plt.plot(K_RANGE, values, color='gray', alpha=0.3, lw=1)

    # Calculate and plot the average inertia across all files (bold red)
    mean_inertia = np.mean(all_inertias, axis=0)
    plt.plot(K_RANGE, mean_inertia, 'ro-', markersize=8, lw=3, label='Mean Inertia')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method: Determining Optimal Number of Strata')
    plt.xticks(K_RANGE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot for your thesis documentation
    plt.savefig('elbow_plot_analysis.png', dpi=300)
    print("\nAnalysis complete. Plot saved as 'elbow_plot_analysis.png'.")
    plt.show()

if __name__ == '__main__':
    run_elbow_analysis()