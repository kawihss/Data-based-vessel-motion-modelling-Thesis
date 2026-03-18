#00_full_preprocessing_pipeline.py
#!/usr/bin/env python3
"""
Executes preprocessing steps 01 to 05 in sequence, clearing output/ directory first for a clean run.
Expects raw AIS data in Data/[Kiel|Bremerhaven|Mississippi]/
"""

import sys
import subprocess
from pathlib import Path

# Pipeline steps (script name, input, output)
STEPS = [
    ("01_extract.py", "Data/", "output/01_raw/"),
    ("02_clean.py", "output/01_raw/", "output/02_cleaned/"),
    ("03_sample.py", "output/02_cleaned/", "output/03_sampled/"),
    ("04_track_segmentation.py", "output/03_sampled/", "output/04_trajectories/"),
    ("05_normalize.py", "output/04_trajectories/", "output/05_normalized/")
]

def run_script(script_name: str):
    try:
        print(f"\n{'='*70}")
        print(f"Running {script_name}")
        print(f"{'='*70}")
        result = subprocess.run([sys.executable, script_name], 
                              check=True, capture_output=False, text=True)
        print(f"{script_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    print("AIS Vessel Trajectory Pipeline")
    print("="*70)
    
    # Step 0: Clear all output directories (non-recursive, only contents)
    print("\nClearing output directories...")
    output_base = Path("output")
    for step_dir in output_base.glob("0*"):  #regex for directories starting with 0 (01_raw, 02_cleaned, etc.)
        if step_dir.is_dir():
            for file in step_dir.glob("*"):
                file.unlink() # delete file
            print(f"   Cleared {step_dir.name}")
    
    # Step 1-5: Run pipeline
    for script_name, _, _ in STEPS:
        run_script(script_name)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("Final data ready in output/05_normalized/")
    print("Scalers saved as output/05_normalized/scalers.pkl")
    print("="*70)
