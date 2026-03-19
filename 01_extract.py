# AIS Data Extractor for Multiple Datasets 
# Supports: Kiel, Bremerhaven (semicolon format) + Marine Cadastre (CSV format)
# Output format: dataset, t_utc, vessel_id, lat, lon, sog, cog, heading, rot
# Optional metadata: vessel_name, imo, call_sign, vessel_type, length, width, draft, cargo, nav_status, true_heading

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_coordinate(coord_str):
    """Parse coordinate string like '54.419327N' or '10.280777E'"""
    coord_str = coord_str.strip()
    if not coord_str or coord_str == 'unk.':
        return np.nan

    direction = coord_str[-1]
    value = float(coord_str[:-1])

    if direction in ['S', 'W']:
        value = -value

    return value


def parse_angle(angle_str):
    """Parse angle string like '279.9°' or '511°' (511 = not available)"""
    angle_str = angle_str.strip().replace('°', '').replace("'", '')
    if angle_str == 'unk.' or not angle_str:
        return np.nan

    value = float(angle_str)
    if value == 511.0: # known AIS sentinel for "not available" heading
        return np.nan

    return value


def parse_speed(speed_str):
    """Parse speed string like '0.0kt'"""
    speed_str = speed_str.strip().replace('kt', '')
    if not speed_str or speed_str == 'unk.':
        return np.nan
    return float(speed_str)


def parse_timestamp(ts_str):
    """Parse timestamp like '210701 000045' -> datetime (YYMMDD format)"""
    ts_str = ts_str.strip()
    try:
        return pd.to_datetime(ts_str, format='%y%m%d %H%M%S')
    except:
        return pd.NaT


def load_ais_data(filepath, dataset_name):
    """
    Load Kiel/Bremerhaven AIS data into pandas DataFrame

    Args:
        filepath: Path to AIS log file
        dataset_name: Name identifier for the dataset (e.g., 'kiel', 'bremerhaven')

    Returns:
        DataFrame with parsed AIS position reports in standard format
    """

    position_records = []
    errors = {'parse_errors': 0, 'invalid_mmsi': 0, 'short_lines': 0}
    line_count = 0

    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    print(f"File: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_count, line in enumerate(f, 1):#  f is an iterable file object

                # Progress update every 50k lines
                if line_count % 50000 == 0:
                    print(f"  Processed {line_count:,} lines | "
                          f"Extracted {len(position_records):,} reports | "
                          f"Errors: {sum(errors.values())}")

                fields = line.strip().split(';') # semicolon delimited for Kiel/Bremerhaven datasets


                # Extract AIS message type from last field
                last_field = fields[-1].strip()
                msg_type_match = last_field.split('[')[-1].rstrip(']')

                # Only process position reports (types 1, 2, 3, 18, 27)
                POSITION_TYPES = {'1', '2', '3', '18', '27'}
                if msg_type_match not in POSITION_TYPES:
                    continue
                
                #after adding the above check, below checks appear to be redundant but we can keep them for extra safety and error tracking 
                # Skip short lines
                if len(fields) < 10:
                    errors['short_lines'] += 1
                    if errors['short_lines'] <= 5:
                        print(f"  Short line {line_count} ({len(fields)} fields): {line.strip()[:200]}")
                    continue
                
                # Check if this is a position report (has coordinates)
                if 'N' not in fields[4] and 'S' not in fields[4]:
                    continue

                # Skip AIS Aids to Navigation (AtoN) MMSI starts with 99 per ITU spec
                if fields[0].strip().startswith('99'):
                    errors['aton_skipped'] = errors.get('aton_skipped', 0) + 1
                    continue
                # Skip SAR aircraft MMSI starts with 111 per ITU spec
                if fields[0].strip().startswith('111'):
                    errors['sar_skipped'] = errors.get('sar_skipped', 0) + 1
                    continue

                try:
                    # Parse MMSI with validation
                    mmsi_str = fields[0].strip()
                    if not mmsi_str.isdigit():
                        errors['invalid_mmsi'] += 1
                        continue

                    record = {
                        'dataset': dataset_name,
                        't_utc': parse_timestamp(fields[9]),
                        'vessel_id': int(mmsi_str),
                        'lat': parse_coordinate(fields[4]),
                        'lon': parse_coordinate(fields[5]),
                        'sog': parse_speed(fields[3]),
                        'cog': parse_angle(fields[6]),
                        'heading': parse_angle(fields[2]),
                        'rot': np.nan,  # Not available
                        'nav_status': fields[1].strip(),
                        'true_heading': parse_angle(fields[7])
                    }
                    position_records.append(record)

                except (ValueError, IndexError) as e:
                    errors['parse_errors'] += 1
                    print(f"  Parse error on line {line_count}: {e} | raw: {line.strip()[:200]}")
                    continue


    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None

    print(f"\nParsing complete!")
    print(f"  Total lines read: {line_count:,}")
    print(f"  Position reports: {len(position_records):,}")
    print(f"  Parse errors: {errors['parse_errors']}")
    print(f"  Invalid MMSI: {errors['invalid_mmsi']}")
    print(f"  Short lines: {errors['short_lines']}")

    # Create DataFrame
    df = pd.DataFrame(position_records)

    if df.empty:
        print("WARNING: No valid position reports found!")
        return df

    # Remove records with invalid timestamps or coordinates
    initial_len = len(df)
    df = df.dropna(subset=['t_utc', 'lat', 'lon'])
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} records with invalid data")

    
    # Convert categorical columns #todo required?
    df['dataset'] = df['dataset'].astype('category')
    df['nav_status'] = df['nav_status'].astype('category')

    # Sort by vessel_id and timestamp
    df = df.sort_values(['vessel_id', 't_utc']).reset_index(drop=True)

    print(f"\nDataset summary:")
    print(f"  Unique vessels: {df['vessel_id'].nunique()}")
    print(f"  Time range: {df['t_utc'].min()} to {df['t_utc'].max()}")
    print(f"  Duration: {df['t_utc'].max() - df['t_utc'].min()}")
    print(f"  Geographic bounds:")
    print(f"    Lat: {df['lat'].min():.4f}°N to {df['lat'].max():.4f}°N")
    print(f"    Lon: {df['lon'].min():.4f}°E to {df['lon'].max():.4f}°E")

    return df


def load_marinecadastre_ais(filepath, dataset_name):
    """
    Load Marine Cadastre AIS CSV data into pandas DataFrame

    Args:
        filepath: Path to Marine Cadastre CSV file
        dataset_name: Name identifier (e.g., 'san_francisco', 'new_york')

    Returns:
        DataFrame with parsed AIS data in standard format
    """

    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} dataset (Marine Cadastre)")
    print(f"{'='*60}")
    print(f"File: {filepath}")

    try:
        df = pd.read_csv(filepath)

        print(f"  Loaded {len(df):,} records")

    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None

    # Check required columns
    required_cols = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return None

    # Rename to match standard format
    df = df.rename(columns={
        'MMSI': 'vessel_id',
        'BaseDateTime': 't_utc',
        'LAT': 'lat',
        'LON': 'lon',
        'SOG': 'sog',
        'COG': 'cog',
        'Heading': 'heading',
        'Status': 'nav_status',
        'VesselName': 'vessel_name',
        'IMO': 'imo',
        'CallSign': 'call_sign',
        'VesselType': 'vessel_type',
        'Length': 'length',
        'Width': 'width',
        'Draft': 'draft',
        'Cargo': 'cargo'
    })

    # Add dataset identifier
    df['dataset'] = dataset_name

    # Parse timestamp
    df['t_utc'] = pd.to_datetime(df['t_utc'], errors='coerce') # unified parser would have unneccesary overhead

    # Add ROT column (not in Marine Cadastre)
    df['rot'] = np.nan

    # Remove invalid records
    initial_len = len(df)
    df = df.dropna(subset=['t_utc', 'lat', 'lon', 'vessel_id'])
    # Fix NOAA Marine Cadastre known encoding errors (per NOAA FAQ)
    # Negative SOG/COG are encoding artifacts, not real negatives
    df.loc[df['sog'] < 0, 'sog'] += 102.4
    df.loc[df['cog'] < 0, 'cog'] += 409.6

    # Sentinel values NaN (not available)
    #df.loc[df['sog'] >= 102.3, 'sog'] = np.nan specs were unclear " Sentinel values such as 102.3 may indicate that no SOG value was
    #transmitted, and can be translated as “not available."
    df.loc[df['cog'] == 360.0, 'cog'] = np.nan
    df.loc[df['heading'] == 511, 'heading'] = np.nan  # standard AIS sentinel

    print(f"  SOG/COG sentinels cleaned")

    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} records with invalid data")

        # Convert categorical columns, again  TODO
    df['dataset'] = df['dataset'].astype('category')
    if 'nav_status' in df.columns:
        df['nav_status'] = df['nav_status'].astype('category')
    if 'vessel_type' in df.columns:
        df['vessel_type'] = df['vessel_type'].astype('category')

    # Sort
    df = df.sort_values(['vessel_id', 't_utc']).reset_index(drop=True)

    print(f"\nDataset summary:")
    print(f"  Unique vessels: {df['vessel_id'].nunique()}")
    print(f"  Time range: {df['t_utc'].min()} to {df['t_utc'].max()}")
    print(f"  Duration: {df['t_utc'].max() - df['t_utc'].min()}")
    print(f"  Geographic bounds:")
    print(f"    Lat: {df['lat'].min():.4f}° to {df['lat'].max():.4f}°")
    print(f"    Lon: {df['lon'].min():.4f}° to {df['lon'].max():.4f}°")

    if 'vessel_type' in df.columns:
        print(f"\n  Vessel types (top 5):")
        type_counts = df['vessel_type'].value_counts().head(5)
        for vtype, count in type_counts.items():
            print(f"    Type {vtype}: {count:,} records")

    return df


# Usage
if __name__ == "__main__":

    # Define datasets
    datasets = [
        {
             'name': 'kiel',
             'file': 'Data/Kiel/ship_emissions_Kiel_AIS_shipdata_20210701.log.txt',
             'output': 'output/01_raw/processed_ais_kiel_20210701.csv',
             'parser': 'kiel'
        },
        #{
        #     'name': 'kiel',
        #     'file': 'Data/Kiel/ship_emissions_Kiel_AIS_shipdata_20250813.log.txt',
        #     'output': 'output/01_raw/processed_ais_kiel_20250813.csv',
        #     'parser': 'kiel'
        #},
        {
             'name': 'bremerhaven',
             'file': 'Data/Bremerhaven/ship_emissions_Bremerhaven_AIS_shipdata_20180404.log.txt',
             'output': 'output/01_raw/processed_ais_bremerhaven_20180404.csv',
             'parser': 'kiel'
        },
                {
             'name': 'wedel',
             'file': 'Data/Wedel/ship_emissions_Wedel_AIS_shipdata_20220405.log.txt',
             'output': 'output/01_raw/processed_Wedel_AIS_shipdata_20220405.csv',
             'parser': 'kiel'
        },
        #{
        #    'name': 'marinecadastre',
        #    'file': 'Data/Mississippi/AIS_2024_01_01.csv',
        #    'output': 'output/01_raw/processed_ais_marinecadastre_2024_01.csv',
        #    'parser': 'marinecadastre'
        #}
    ]

    # Create output directory
    Path('output/01_raw').mkdir(parents=True, exist_ok=True)

    all_data = []

    # Process each dataset
    for dataset_config in datasets:
        # Select appropriate parser
        if dataset_config.get('parser') == 'marinecadastre':
            df = load_marinecadastre_ais(
                filepath=dataset_config['file'],
                dataset_name=dataset_config['name'],
            )
        else:
            # Default: Kiel/Bremerhaven format
            df = load_ais_data(
                filepath=dataset_config['file'],
                dataset_name=dataset_config['name'],
            )

        if df is not None and not df.empty:
            # Define core columns (always present)
            core_columns = ['dataset', 't_utc', 'vessel_id', 'lat', 'lon', # 'x', 'y', 
                          'sog', 'cog', 'heading', 'rot']

            # Define optional columns (may or may not be present)
            optional_columns = ['nav_status', 'true_heading', 'vessel_name', 'imo', 
                              'call_sign', 'vessel_type', 'length', 'width', 'draft', 'cargo']

            # Reorder: core first, then optional (only if present)
            column_order = core_columns + [col for col in optional_columns if col in df.columns]
            df = df[[col for col in column_order if col in df.columns]]

            # Save individual dataset
            df.to_csv(dataset_config['output'], index=False)
            print(f"Saved to: {dataset_config['output']}")
            print(f"Records: {len(df):,} from {df['vessel_id'].nunique()} vessels")


        else:
            print(f"✗ Failed to process {dataset_config['name']}")

    """
    # Combine all datasets
    print(f"\n{'='*60}")
    print("COMBINING ALL DATASETS")
    print(f"{'='*60}")

    # Concatenate (pandas handles missing optional columns with NaN)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['dataset', 'vessel_id', 't_utc']).reset_index(drop=True)

    combined_output = 'output/01_raw/processed_ais_combined.csv'
    combined_df.to_csv(combined_output, index=False)

    print(f"\nCombined dataset statistics:")
    print(f"  Total records: {len(combined_df):,}")
    print(f"  Total unique vessels: {combined_df['vessel_id'].nunique()}")
    print(f"  Datasets: {combined_df['dataset'].unique().tolist()}")
    print(f"\nRecords per dataset:")
    print(combined_df.groupby('dataset', observed=True).size())
    print(f"\n Saved combined data to: {combined_output}")

    print(f"\n{'='*60}")
    print(f"OUTPUT FORMAT:")
    print(f"{'='*60}")
    print(f"Core columns: dataset, t_utc, vessel_id, lat, lon, sog, cog, heading, rot")
    print(f"Optional columns: {[col for col in combined_df.columns if col not in ['dataset', 't_utc', 'vessel_id', 'lat', 'lon', 'sog', 'cog', 'heading', 'rot']]}")

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    """
