import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models.kinematic import ConstantVelocityModel
from evaluation.evaluator import run_evaluation, plot_horizon_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Evaluate ConstantVelocityModel on all test files
    project_root = Path(__file__).resolve().parent.parent
    model = ConstantVelocityModel(velocity_fraction=0.4)
    metrics = run_evaluation(model, project_root / "output/05_normalized", split='test')
    print(metrics)

    per_file_metrics = metrics.get('per_file_metrics', pd.DataFrame())
    per_month_metrics = metrics.get('per_month_metrics', pd.DataFrame())

    diagnostics_dir = project_root / "evaluation" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if not per_file_metrics.empty:
        per_file_path = diagnostics_dir / "test_metrics_per_file.csv"
        per_file_metrics.to_csv(per_file_path, index=False)
        print("\nWorst 10 files by RMSE:")
        print(per_file_metrics.head(10).to_string(index=False))
        print(f"Saved per-file metrics to {per_file_path}")

    if not per_month_metrics.empty:
        per_month_path = diagnostics_dir / "test_metrics_per_month.csv"
        per_month_metrics.to_csv(per_month_path, index=False)
        print("\nMonths sorted by RMSE:")
        print(per_month_metrics.to_string(index=False))
        print(f"Saved per-month metrics to {per_month_path}")

    # Plot ADE(t)
    fig, ax = plot_horizon_error(metrics, label="Constant Velocity")
    plt.show()
"""
Streaming 1968 file(s) for split 'test'
{'ADE': 151.1338190980579, 'FDE': 472.50075887533666, 'RMSE': 78956.52034362784, 'ADE_per_step': array([ 12.97453588,  28.06489758,  46.22589108,  66.87948312,
        90.23111604, 117.69888618, 156.99708102, 187.42300146,
       332.34253974, 472.50075888]), 'n_tracks': 1010821}
"""
"""Streaming 1887 file(s) for split 'test'
{'ADE': 92.74923321505018, 'FDE': 200.20702327876316, 'RMSE': 178.43278622724145, 'ADE_per_step': array([ 10.73234567,  22.71583283,  37.46408122,  54.37784362,
        73.67363583,  95.00596971, 118.42751695, 143.79687242,
       171.09121062, 200.20702328]), 'n_tracks': 925583, 'per_file_metrics': """#fixed by adjusting time gap guard to 1.