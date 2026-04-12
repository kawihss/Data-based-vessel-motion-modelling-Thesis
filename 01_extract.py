# AIS Data Extractor 

# Output format: dataset, t_utc, vessel_id, lat, lon, sog, cog, heading, rot

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

TEST_LIMIT =  2; None #5  # Limit number of files per dataset for testing. None = process all files. 

def parse_coordinate(coord_str):
    coord_str = coord_str.strip()
    if not coord_str or coord_str == 'unk.':
        return np.nan

    direction = coord_str[-1]
    value = float(coord_str[:-1])

    if direction in ['S', 'W']:
        value = -value

    return value


def parse_angle(angle_str):
    angle_str = angle_str.strip().replace('°', '').replace("'", '')
    if angle_str == 'unk.' or not angle_str:
        return np.nan

    value = float(angle_str)
    if value == 511.0: # known AIS sentinel for "not available" heading
        return np.nan

    return value


def parse_speed(speed_str):
    speed_str = speed_str.strip().replace('kt', '')
    if not speed_str or speed_str == 'unk.':
        return np.nan
    return float(speed_str)


def parse_timestamp(ts_str):
    ts_str = ts_str.strip()
    try:
        return pd.to_datetime(ts_str, format='%y%m%d %H%M%S')
    except:
        return pd.NaT


def load_ais_data(filepath, dataset_name):
    #Load parsed AIS data into pandas DataFrame

    #dataset_name: Name identifier for the dataset (e.g., 'kiel', 'bremerhaven')
    

    position_records = []
    errors = {'parse_errors': 0, 'invalid_mmsi': 0, 'short_lines': 0}
    line_count = 0


    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_count, line in enumerate(f, 1):

                fields = line.strip().split(';') # semicolon delimited for Kiel/Bremerhaven datasets


                # Extract AIS message type from last field
                last_field = fields[-1].strip()
                msg_type_match = last_field.split('[')[-1].rstrip(']')

                # Only process position reports (types 1, 2, 3, 18, 27)
                POSITION_TYPES = {'1', '2', '3', '18', '27'}
                if msg_type_match not in POSITION_TYPES:
                    continue
                
                #after adding the above check, below checks appear to be redundant but we can keep them for extra safety and error tracking 
                #watching the preprocessing resulted in 0 for all these errors

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
                        #'t_utc': parse_timestamp(fields[9]), # slower, doing all at once after creating DataFrame
                        't_utc': fields[9].strip(), 
                        'vessel_id': int(mmsi_str),
                        'lat': parse_coordinate(fields[4]),
                        'lon': parse_coordinate(fields[5]),
                        'sog': parse_speed(fields[3]),
                        'cog': parse_angle(fields[6]),
                        'heading': parse_angle(fields[2]),
                        'rot': np.nan,  # Not available, calculated later in the pipeline
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

    #print(f"\nParsing complete!")
    #print(f"  Total lines read: {line_count:,}")
    #print(f"  Position reports: {len(position_records):,}")
    #print(f"  Parse errors: {errors['parse_errors']}")
    #print(f"  Invalid MMSI: {errors['invalid_mmsi']}")
    #print(f"  Short lines: {errors['short_lines']}")

    # Create DataFrame
    df = pd.DataFrame(position_records)

    if df.empty:
        print("WARNING: No valid position reports found!")
        return df
    
    df['t_utc'] = pd.to_datetime(df['t_utc'], format='%y%m%d %H%M%S', errors='coerce') # converte all at once

    # Remove records with invalid timestamps or coordinates
    #initial_len = len(df)
    df = df.dropna(subset=['t_utc', 'lat', 'lon', 'sog', 'cog'])# added sog and cog after crash in 04. did not include optional features like true_heading
    #if len(df) < initial_len:
    #    print(f"  Removed {initial_len - len(df)} records with invalid data")

    
    df = df.sort_values(['vessel_id', 't_utc']).reset_index(drop=True)

    #print(f"\nDataset summary:")
    print(f"  Unique vessels: {df['vessel_id'].nunique()}")
    #print(f"  Time range: {df['t_utc'].min()} to {df['t_utc'].max()}")
    #print(f"  Duration: {df['t_utc'].max() - df['t_utc'].min()}")
    #print(f"  Geographic bounds:")
    #print(f"    Lat: {df['lat'].min():.4f}°N to {df['lat'].max():.4f}°N")
    #print(f"    Lon: {df['lon'].min():.4f}°E to {df['lon'].max():.4f}°E")

    return df


  

# Usage
if __name__ == "__main__":

    dataset_dirs = {
        'kiel': Path('Data/Kiel'),
        'bremerhaven': Path('Data/Bremerhaven'),
        'wedel': Path('Data/Wedel')
    }

    output_dir = Path('output/01_raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Core columns to ensure consistent structure
    core_columns = ['dataset', 't_utc', 'vessel_id', 'lat', 'lon', 'sog', 'cog', 'heading', 'rot']
    optional_columns = ['nav_status', 'true_heading', 'vessel_name', 'imo', 'call_sign', 'vessel_type', 'length', 'width', 'draft', 'cargo'] # legacy, Marinecadastre dataset

    for dataset_name, data_path in dataset_dirs.items():
        if not data_path.exists():
            print(f"Directory not found: {data_path}")
            continue

        # Find all .log files downloaded from GovData 
        valid_files = list(data_path.glob("*.log")) 
        if TEST_LIMIT:
            valid_files = valid_files[:TEST_LIMIT]


        if not valid_files:
            print(f"No files found in {data_path}")
            continue
            
        for file_path in valid_files:
            # Create unique output name based on original file
            out_filename = f"processed_{dataset_name}_{file_path.stem}.csv"
            out_path = output_dir / out_filename

            if out_path.exists():
                print(f"Skipping {file_path.name}, output already exists.")
                continue

            df = load_ais_data(filepath=file_path, dataset_name=dataset_name)

            if df is not None and not df.empty:
                column_order = core_columns + [col for col in optional_columns if col in df.columns]
                df = df[[col for col in column_order if col in df.columns]]

                df.to_csv(out_path, index=False)
                print(f"Saved to: {out_path}")
                print(f"Records: {len(df):,} from {df['vessel_id'].nunique()} vessels\n")