# Visualize AIS trajectory context assignments (harbour, river, channel, lock)
# Input:  output/02_cleaned/*.csv

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_context_assignments(csv_path: Path):
    """
    Plots AIS trajectory points colored by their assigned geographic context.
    Uses relative UTM coordinates (meters) to avoid spatial distortion.
    """
    print(f"Loading data: {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # 1. Define color scheme
    color_map = {
        'harbour': '#7f7f7f',   # Grey
        'unknown': '#CC43CE',   # Magenta (out of bounds / error)
        'river':   '#1f77b4',   # Blue
        'channel': '#2ca02c',   # Green
        'lock':    '#d62728'    # Red
    }
    
    # 2. Define rendering order (z-order), opacity (alpha), and point size (s)
    # Background/frequent contexts are rendered first, specific areas (locks) last
    plot_order = [
        {'label': 'unknown', 'z': 1, 'alpha': 0.1, 's': 1},
        {'label': 'harbour', 'z': 2, 'alpha': 0.3, 's': 1},
        {'label': 'river',   'z': 3, 'alpha': 0.6, 's': 2},
        {'label': 'channel', 'z': 4, 'alpha': 0.8, 's': 2},
        {'label': 'lock',    'z': 5, 'alpha': 1.0, 's': 15}  # Emphasize locks
    ]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#f0f0f0')  # Light grey background to contrast missing data areas
    
    # 3. Scatter plot for each context group
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
            
    # Formatting
    ax.set_title(f"Context Assignments: {csv_path.name}\n(Coordinates in relative meters)", fontsize=14)
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    
    # Standardize legend marker sizes
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
        # Plot the first file found
        plot_context_assignments(files[2])
