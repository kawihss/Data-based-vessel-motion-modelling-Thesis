import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd

def fetch_and_plot_water_tags(place_name="Kiel, Germany"):

    print(f"Fetching OSM water features for: {place_name}...")
    
    tags = {
        "natural": "water",
        "waterway": True,
        "landuse": ["harbour", "port"],
        "harbour": True,
        "leisure": "marina",
        "bay": True
    }

    try:
        gdf = ox.features_from_place(place_name, tags)
    except Exception as e:
        print(f"Failed to fetch data from OSM: {e}")
        return

    def assign_label(row):
        if 'waterway' in row and pd.notna(row['waterway']):
            return f"waterway={row['waterway']}"
        if 'water' in row and pd.notna(row['water']):
            return f"water={row['water']}"
        if 'landuse' in row and pd.notna(row['landuse']):
            return f"landuse={row['landuse']}"
        
        return "other tags"

    gdf['plot_label'] = gdf.apply(assign_label, axis=1)

    fig, ax = plt.subplots(figsize=(15, 12))
    
    gdf.plot(
        ax=ax, 
        column='plot_label', 
        legend=True, 
        legend_kwds={'bbox_to_anchor': (1, 1)}, 
        cmap='tab20', 
        alpha=0.7
    )
    
    ax.set_title(f"Overview of OSM Tags: {place_name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fetch_and_plot_water_tags("Kiel, Germany")