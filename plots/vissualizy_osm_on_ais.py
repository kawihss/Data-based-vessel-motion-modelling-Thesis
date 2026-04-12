# Visualize AIS trajectory context assignments (harbour, river, channel, lock)
# Input:  output/02_cleaned/*.csv

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_context_assignments(csv_path: Path):

    df = pd.read_csv(csv_path)
    
    color_map = {
        'harbour': '#7f7f7f',   # Grey
        'unknown': '#CC43CE',   # Magenta (out of bounds / error)
        'river':   '#1f77b4',   # Blue
        'channel': '#2ca02c',   # Green
        'lock':    '#d62728'    # Red
    }
    
    # rendering order (z-order), opacity (alpha), and point size (s)
    plot_order = [
        {'label': 'unknown', 'z': 1, 'alpha': 0.1, 's': 1},
        {'label': 'harbour', 'z': 2, 'alpha': 0.3, 's': 1},
        {'label': 'channel', 'z': 3, 'alpha': 0.8, 's': 2},
        {'label': 'river',   'z': 4, 'alpha': 0.6, 's': 2},
        {'label': 'lock',    'z': 5, 'alpha': 1.0, 's': 15}  # Emphasize locks
    ]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#f0f0f0') 
    
    for config in plot_order:
        label = config['label']
        group = df[df['context'] == label]
        
        if not group.empty:
            ax.scatter(
                group['x'], group['y'], 
                c=color_map[label], 
                label=f"{label} (n={len(group):,})",
                alpha=config['alpha'],
                s=config['s'],
                zorder=config['z'],
                edgecolors='none'
            )
            
    ax.set_title(f"Context Assignments: {csv_path.name}\n(Coordinates in relative meters)", fontsize=14)
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    
    lgnd = ax.legend(title="Context", loc="upper right")
    for handle in lgnd.legend_handles:
        handle.set_sizes([50])
        handle.set_alpha(1.0)
        
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cleaned_dir = Path("output/02_cleaned")
    files = list(cleaned_dir.glob("*.csv"))
    
    if not files:
        print(f"No cleaned CSV files found in {cleaned_dir}")
    else:
        for csv_file in files:
            plot_context_assignments(csv_file)
