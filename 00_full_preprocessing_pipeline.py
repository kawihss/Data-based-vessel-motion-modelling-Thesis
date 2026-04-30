#Executes preprocessing steps 01 to 05 in sequence, clearing output/ directory first for a clean run.
#Input: raw AIS data in Data/[Kiel|Bremerhaven|Wedel]/

import sys
import subprocess
import time
from pathlib import Path


START_AT_STEP = 2   # Choose 1-6
CLEAR_OUTPUTS = True # Clears folders for active steps

STEPS = [
    ("01_extract.py", "Data/", "output/01_raw/"),
    ("02_clean.py", "output/01_raw/", "output/02_cleaned/"),
    ("03_sample.py", "output/02_cleaned/", "output/03_sampled/"),
    ("04_track_segmentation.py", "output/03_sampled/", "output/04_trajectories/"),
    ("05_normalize.py", "output/04_trajectories/", "output/05_normalized/"),
    ("07_generate_parquet.py", "output/05_normalized/", "output/07_parquet/")
]

def run_script(script_name: str):
    try:
        print(f"Running {script_name}")
        start_time = time.time()
        subprocess.run([sys.executable, script_name], check=True)
        duration = time.time() - start_time
        print(f"{script_name} completed successfully")
        return duration
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    print("AIS Vessel Trajectory Pipeline")
    print(f"Resume from Step: {START_AT_STEP} | Clear: {CLEAR_OUTPUTS}")
    
    output_base = Path("output")
    
    # Step 0: Optional Cleanup
    if CLEAR_OUTPUTS:
        for i in range(START_AT_STEP - 1, len(STEPS)):
            step_output = Path(STEPS[i][2])
            if step_output.is_dir():
                for file in step_output.glob("*"):
                    file.unlink()
                print(f"   Cleared {step_output.name}")
    
    timings = {}
    for i in range(START_AT_STEP - 1, len(STEPS)):
        script_name, _, _ = STEPS[i]
        timings[script_name] = run_script(script_name)
    
    print("PIPELINE COMPLETE!")
    
    for script, duration in timings.items():
        print(f"{script}: {duration:.2f} seconds")