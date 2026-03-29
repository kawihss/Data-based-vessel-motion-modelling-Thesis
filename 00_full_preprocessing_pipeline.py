#00_full_preprocessing_pipeline.py
#!/usr/bin/env python3
"""
Executes preprocessing steps 01 to 05 in sequence, clearing output/ directory first for a clean run.
Expects raw AIS data in Data/[Kiel|Bremerhaven|Mississippi]/
"""

#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path


START_AT_STEP = 2   # Choose 1-5
CLEAR_OUTPUTS = False # Only clears folders for active steps

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
        subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    print("AIS Vessel Trajectory Pipeline")
    print(f"Resume from Step: {START_AT_STEP} | Clear: {CLEAR_OUTPUTS}")
    print("="*70)
    
    output_base = Path("output")
    
    # Step 0: Optional Cleanup
    if CLEAR_OUTPUTS:
        print("\nClearing output directories...")
        for i in range(START_AT_STEP, len(STEPS) + 1):
            for step_dir in output_base.glob(f"0{i}*"):# directories starting with 01, 02, etc.
                if step_dir.is_dir():
                    for file in step_dir.glob("*"):
                        file.unlink()
                    print(f"   Cleared {step_dir.name}")
    
    for i in range(START_AT_STEP - 1, len(STEPS)):
        script_name, _, _ = STEPS[i]
        run_script(script_name)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)