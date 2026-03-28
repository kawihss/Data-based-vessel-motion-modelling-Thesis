import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
ox.settings.max_query_area_size = 5 * 1e9  
ox.settings.timeout = 600



def get_water_features(place_name="Kiel, Germany"):
    """
    Fetches and categorizes OSM water features into river, channel, lock, and other.
    """
    tags = {
        "natural": "water",      
        "waterway": True,        
        "landuse": ["harbour", "port"], 
        "harbour": True,         
        "leisure": "marina",     
        "bay": True,
        "lock": True             # Explicitly fetch features tagged as locks
    }

    try:
        gdf = ox.features_from_place(place_name, tags)
    except Exception as e:
        print(f"Error fetching OSM data for {place_name}: {e}")
        return None

    def categorize(row):
        waterway = str(row.get('waterway', '')).lower()
        water = str(row.get('water', '')).lower()
        lock = str(row.get('lock', '')).lower()

        # 1. Lock (Schleuse) - Highest priority, overrides river/channel
        if waterway == 'lock' or lock in ['yes', 'true', '1']:
            return 'lock'

        # 2. River
        if waterway == 'river' or water == 'river':
            return 'river'
        
        # 3. Channel
        if waterway == 'canal' or water in ['canal', 'channel']:
            return 'channel'
        
        # 4. Remove unwanted small features
        unwanted = {'drain', 'ditch', 'stream', 'swimming_pool', 'reflecting_pool', 'wastewater'}
        if waterway in unwanted or water in unwanted:
            return 'remove'
        
        # 5. Other (Harbours, bays, unsanitized water)
        return 'other'

    gdf['water_class'] = gdf.apply(categorize, axis=1)
    
    # Filter out unwanted tags and isolated point geometries
    gdf = gdf[gdf['water_class'] != 'remove']
    valid_geoms = {'Polygon', 'MultiPolygon', 'LineString', 'MultiLineString'}
    gdf = gdf[gdf.geometry.type.isin(valid_geoms)].copy()

    return gdf

def plot_water_features(gdf, place_name):
    """Plots the categorized water features with a custom legend."""
    fig, ax = plt.subplots(figsize=(15, 12))
    
    color_map = {
        'river': '#1f77b4',     # Blue
        'channel': '#2ca02c',   # Green
        'lock': '#d62728',      # Red
        'other': '#7f7f7f'      # Gray
    }
    
    for label, group in gdf.groupby('water_class'):
        group.plot(ax=ax, color=color_map.get(label, 'black'), alpha=0.8)
    
    handles = [mpatches.Patch(color=c, label=l, alpha=0.8) for l, c in color_map.items()]
    ax.legend(handles=handles, title="Water Category", bbox_to_anchor=(1, 1))
    
    ax.set_title(f"OSM Water Categories: {place_name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    place = "Wedel, Germany"  
    print(f"Processing {place}...")
    
    gdf_water = get_water_features(place)
    if gdf_water is not None:
        plot_water_features(gdf_water, place)