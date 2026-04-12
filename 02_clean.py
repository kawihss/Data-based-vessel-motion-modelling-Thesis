# AIS Cleaning: coordinate conversion, deduplication, MMSI hashing, context labeling
# Input:  output/01_raw/*.csv
# Output: output/02_cleaned/*.csv this is what we can publish, 01_raw is not anonymized yet, features are output later

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime
from pyproj import Transformer
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

#  OSMnx API Settings
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter" # mirror that is more reliable for large queries
ox.settings.max_query_area_size = 5 * 1e9  # 5000 km² 
ox.settings.timeout = 600                  # 10 minutes timeout for large queries


# HA-256 Hashing to 12 characters
def hash_mmsi(mmsi) -> str:
    return hashlib.sha256(str(int(mmsi)).encode()).hexdigest()[:12]


def add_utm_coordinates(df: pd.DataFrame):
    df = df.copy()
    median_lon = df['lon'].median()
    median_lat = df['lat'].median()
    zone = int((median_lon + 180) / 6) + 1
    epsg = f"326{zone:02d}" if median_lat >= 0 else f"327{zone:02d}"#epsg code for UTM zone, 326 for northern hemisphere, 327 for southern hemisphere

    #Transformer is a class from pyproj that handles coordinate transformations. 
    # We create a transformer object that converts from WGS84 (EPSG:4326) to the appropriate UTM zone based on the median coordinates of the dataset.
    # always_xy=True ensures that the input order is (lon, lat) which is required for pyproj.
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True) 

    valid = df['lat'].notna() & df['lon'].notna()
    x = np.full(len(df), np.nan)
    y = np.full(len(df), np.nan)
    x[valid], y[valid] = transformer.transform(
        df.loc[valid, 'lon'].values,
        df.loc[valid, 'lat'].values
    )

    df['x'] = x
    df['y'] = y

    hemi = 'N' if median_lat >= 0 else 'S'
    return df, epsg

def fetch_osm_water_features(west, south, east, north) -> gpd.GeoDataFrame:
    tags = {
        "natural": "water",      
        "waterway": True,        
        "landuse": ["harbour", "port"], 
        "harbour": True,         
        "leisure": "marina",     
        "bay": True,
        "lock": True
    }
    
    try:
        # Create a polygon from the bounding box and fetch OSM features within it
        bounding_polygon = box(west, south, east, north)
        gdf = ox.features_from_polygon(bounding_polygon, tags)
    except Exception as e:
        print(f"  WARNING: Failed to fetch OSM data: {e}")
        return None

    if gdf.empty:
        return None

    def categorize(row): # same logic as in assign_relevant_waterway
        waterway = str(row.get('waterway', '')).lower()
        water = str(row.get('water', '')).lower()
        lock = str(row.get('lock', '')).lower()

        if waterway == 'lock' or water == 'lock' or lock in ['yes', 'true', '1']:
            return 'lock'
        if waterway == 'river' or water == 'river':
            return 'river'
        if waterway == 'canal' or water in ['canal', 'channel']:
            return 'channel'
        
        unwanted = {'drain', 'ditch', 'stream', 'swimming_pool', 'reflecting_pool', 'wastewater'}
        if waterway in unwanted or water in unwanted:
            return 'remove'
        return 'harbour'

    gdf['water_class'] = gdf.apply(categorize, axis=1)
    gdf = gdf[gdf['water_class'] != 'remove']
    
    valid_geoms = {'Polygon', 'MultiPolygon', 'LineString', 'MultiLineString'}
    return gdf[gdf.geometry.type.isin(valid_geoms)].copy()

def assign_water_context(df: pd.DataFrame, epsg: str, dataset_name: str) -> pd.DataFrame:

#  Static boxes are very big, but cover all files after filter -> cheaper than downloading smaller boxes for each file
    if 'bremerhaven' in dataset_name:
        south, north = 53.40, 54.00
        west, east = 8.00, 8.80
    elif 'kiel' in dataset_name:
        south, north = 53.50, 54.50
        west, east = 9.20, 11.50
    elif 'wedel' in dataset_name:
        south, north = 53.00, 54.00
        west, east = 9.00, 10.50
    else:
        # Fallback uses outer bound of file. this is slower because we fetch from osmnx every file
        pad = 0.05
        south = df['lat'].min() - pad
        north = df['lat'].max() + pad
        west = df['lon'].min() - pad
        east = df['lon'].max() + pad


    osm_gdf = fetch_osm_water_features(west, south, east, north)
    
    if osm_gdf is None or osm_gdf.empty:
        print("  WARNING: No OSM data found in this area.")
        df['context'] = 'unknown'
        return df

    osm_gdf = osm_gdf.to_crs(f"EPSG:{epsg}")

 

    if 'bremerhaven' in dataset_name:
        buffer_radii = {
            'lock': 50,       
            'channel': 0,     # no channel that should be assigned in bremerhaven
            'river': 250,     
            'harbour': 800    
        }
    elif 'kiel' in dataset_name:
        buffer_radii = {
            'lock': 50,       
            'channel': 50,    # Nord-Ostsee-Kanal
            'river': 250,     
            'harbour': 1000   
        }
    else: # Default (Wedel)
        buffer_radii = {
            'lock': 0,     # no locks in Wedel 
            'channel': 50,    
            'river': 250,     
            'harbour': 0 # There are harbours in Wedel but they are small and close to the river, no harbour in our sense    
            #having a buffer as big as in the others would include outliers (see pink points in visualize_osm_on_ais.py in wedel)
        }
    
    osm_gdf['geometry'] = osm_gdf.apply(
        lambda row: row.geometry.buffer(buffer_radii.get(row.water_class, 50)), 
        axis=1
    )

    gdf_points = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.x, df.y), 
        crs=f"EPSG:{epsg}"
    )

    joined = gpd.sjoin(
        gdf_points, 
        osm_gdf[['water_class', 'geometry']], 
        how='left', 
        predicate='within'
    )

    priority_map = {'lock': 1, 'river': 2, 'channel': 3, 'harbour': 4, 'unknown': 5}#todo in thesis reihenfolge anpassen
    
    joined['point_id'] = joined.index
    joined['priority'] = joined['water_class'].map(priority_map).fillna(99)
    
    joined = joined.sort_values(['point_id', 'priority'])
    
    #keep only the highest priority context for each point_id
    joined = joined[~joined['point_id'].duplicated(keep='first')]
    
    df['context'] = joined['water_class'].fillna('unknown')
    
    counts = df['context'].value_counts().to_dict()
    print(f"  Context labels assigned: {counts}")
    
    return df


def remove_position_jumps(df: pd.DataFrame, threshold=2.0) -> pd.DataFrame:
    #Delete rows where displacement exceeds threshold × speed-implied distance
    df = df.copy().sort_values('t_utc').reset_index(drop=True)

    dt = df['t_utc'].diff().dt.total_seconds().values # (works on all Pandas versions). carefull, timestamp unit changes between pandas versions!!
    dx = np.diff(df['x'].values, prepend=np.nan)
    dy = np.diff(df['y'].values, prepend=np.nan)
    actual_dist = np.sqrt(dx**2 + dy**2)             # metres

    speed_ms = df['sog'].values * 1852 / 3600        #  knot to m/s
    speed_avg = (speed_ms + np.roll(speed_ms, 1)) / 2
    speed_avg[0] = speed_ms[0]
    expected_dist = speed_avg * np.abs(dt)

    is_jump = actual_dist > threshold * np.maximum(expected_dist, 10) # also set a minimum expected distance to avoid flagging small time gaps at low speeds
    is_jump[0] = False

    return df[~is_jump].reset_index(drop=True)

def remove_fast_vessels(df: pd.DataFrame, max_sog=30) -> pd.DataFrame:
    #Remove all rows belonging to vessels that ever exceed max_sog knots
    fast_ids = df.groupby('vessel_id')['sog'].max()
    fast_ids = fast_ids[fast_ids > max_sog].index
    mask = df['vessel_id'].isin(fast_ids)
    print(f"  Fast vessel filter: removed {mask.sum():,} rows "
            f"({fast_ids.nunique()} vessels exceeding {max_sog} kn)")
    return df[~mask].reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # Geographic bounds filter. very vague, just to catch obvious outliers and coordinate errors. 
    n_pre_geo = len(df)  

    dataset_name = df['dataset'].iloc[0].lower()
    if 'bremerhaven' in dataset_name:
        mask = df['lat'].between(53.40, 54) & df['lon'].between(8, 8.8)
    elif 'kiel' in dataset_name:
        mask = df['lat'].between(53.5, 54.5) & df['lon'].between(9.2, 11.5)
    elif 'wedel' in dataset_name:
        mask = df['lat'].between(53, 54) & df['lon'].between(9, 10.5)
    else:
        raise ValueError(f"Unknown dataset for geographic bounds filter: {dataset_name}")

    df = df[mask].copy()
    #print(f"  Geographic filter kept {len(df):,} / {n_pre_geo:,} rows")


    # Anonymize MMSI hashed vessel_id
    df['vessel_id'] = df['vessel_id'].apply(hash_mmsi)
    #print(f"  Hashed {df['vessel_id'].nunique()} unique vessel IDs")

    
    # Remove duplicates: keep row with most non-null values
    n_before_dedup = len(df)

    df['_non_null'] = df.notna().sum(axis=1)
    df = df.sort_values('_non_null', ascending=False)
    df = df.drop_duplicates(subset=['vessel_id', 't_utc'], keep='first')
    df = df.drop(columns='_non_null')

    #print(f"  Duplicates removed: {n_before_dedup - len(df):,}")

    # UTM coordinate conversion
    df, epsg = add_utm_coordinates(df)
    
    # assign context labels
    df = assign_water_context(df, epsg, dataset_name)

    # Origin subtraction for positional anonymization (Moved from add_utm_coordinates)
    origin_x = np.nanmedian(df['x'])
    origin_y = np.nanmedian(df['y'])
    df['x'] = df['x'] - origin_x
    df['y'] = df['y'] - origin_y

    # Reorder columns: insert x, y, context after lon
    cols = list(df.columns)
    if 'x' in cols and 'y' in cols:
        cols = [c for c in cols if c not in ('x', 'y', 'context')]
        lon_idx = cols.index('lon')
        cols = cols[:lon_idx+1] + ['x', 'y', 'context'] + cols[lon_idx+1:]
        df = df[cols]

    #Remove position jumps per vessel and filter out vessels that exceed speed threshold 
    n_before_jumps = len(df)
    df = pd.concat([remove_position_jumps(group) for _, group in df.groupby('vessel_id')]).reset_index(drop=True) # for loop because pandas just causes trouble between different versions
    print(f"  Jump filter removed: {n_before_jumps - len(df):,}") # left this in because of the error described in the thesis

    df = remove_fast_vessels(df, max_sog=30)

    #print(f" removed {n0 - len(df):,} rows in total")


    df = df.drop(columns=['lat', 'lon'])

    return df.sort_values(['vessel_id', 't_utc']).reset_index(drop=True)


if __name__ == "__main__":
    input_dir  = Path("output/01_raw")
    output_dir = Path("output/02_cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {input_dir}")
        exit(1)

    total_in = total_out = 0

    for src in input_files:
        print(f"Cleaning: {src.name}")

        df = pd.read_csv(src, parse_dates=['t_utc']).copy()
        n_in = len(df)

        df = clean(df)
        n_out = len(df)

        dst = output_dir / src.name
        df.to_csv(dst, index=False)
        print(f"Saved: {dst}")

        total_in  += n_in
        total_out += n_out

    
    print(f"DONE  {total_in:,} -> {total_out:,} rows")

