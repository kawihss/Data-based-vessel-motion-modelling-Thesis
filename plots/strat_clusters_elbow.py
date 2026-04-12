import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_DIR = Path('output/03_sampled')
K_RANGE = range(2, 16) 

def run_elbow_analysis():
    input_files = sorted(INPUT_DIR.glob('*.csv'))

    all_inertias = []
    file_names = []

    for src in input_files:
        df = pd.read_csv(src)
        
        # Same logic as in main script 
        vessel_profiles = df.groupby('vessel_id').agg({
            'sog': 'mean',
            'rot': lambda x: x.abs().max(), # Max absolute ROT for maneuver intensity
            'context': lambda x: x.mode()[0]
        }).reset_index()

        vessel_profiles['context_idx'] = vessel_profiles['context'].astype('category').cat.codes
        
        scaler = StandardScaler()
        features = vessel_profiles[['sog', 'rot', 'context_idx']]
        scaled_features = scaler.fit_transform(features)

        current_file_inertia = []
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_features)
            current_file_inertia.append(km.inertia_)
        
        all_inertias.append(current_file_inertia)
        file_names.append(src.name)

    plt.figure(figsize=(10, 6))

    for i, values in enumerate(all_inertias):
        plt.plot(K_RANGE, values, color='gray', alpha=0.3, lw=1)

    mean_inertia = np.mean(all_inertias, axis=0)
    plt.plot(K_RANGE, mean_inertia, 'ro-', markersize=8, lw=3, label='Mean Inertia')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method: Determining Optimal Number of Strata')
    plt.xticks(K_RANGE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('elbow_plot_analysis.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    run_elbow_analysis()