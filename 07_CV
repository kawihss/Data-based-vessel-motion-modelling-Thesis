import pandas as pd
import numpy as np
import glob
import os

out_dir = "output/07_baselines/CV"
os.makedirs(out_dir, exist_ok=True)

PRED_STEPS = 10 
DT = 30 

test_files = glob.glob("output/05_normalized/test_*.csv")
all_predictions = []

for f in test_files:
    df = pd.read_csv(f)
    
    for tid, track in df.groupby('track_id'):
        ctx = track[track['role'] == 'context']
        if ctx.empty: continue
        
        # Get last known physical state
        last = ctx.iloc[-1]
        x_now, y_now = last['x'], last['y']
        sog_now, cog_now = last['sog'], last['cog']
        
        # Calculate the constant displacement per step
        v_mps = sog_now * 0.51444
        rad = np.deg2rad(cog_now)
        dx_step = v_mps * np.sin(rad) * DT
        dy_step = v_mps * np.cos(rad) * DT
        
        for step in range(1, PRED_STEPS + 1):
            x_now += dx_step
            y_now += dy_step
            
            all_predictions.append({
                'track_id': tid,
                'step': step,
                'x_pred': x_now,
                'y_pred': y_now,
                'dx_pred': dx_step,
                'dy_pred': dy_step,
                'sog_pred': sog_now,
                'cog_pred': cog_now,
                'context': last['context']
            })

# Save the raw predictions
pred_df = pd.DataFrame(all_predictions)
pred_df.to_csv(f"{out_dir}/cv_predictions.csv", index=False)
print(f"Generated {len(pred_df)} prediction points for {pred_df['track_id'].nunique()} tracks.")