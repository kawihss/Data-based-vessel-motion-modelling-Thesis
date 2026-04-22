import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel
from evaluation.evaluator import run_evaluation, plot_horizon_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    diagnostics_dir = project_root / "evaluation" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("constant_velocity", "Constant Velocity", ConstantVelocityModel(velocity_fraction=0.4)),
        ("ctrv", "CTRV", ConstantTurnRateVelocityModel(velocity_fraction=0.4)),
    ]

    comparison_rows = []
    fig = None
    ax = None

    for model_key, model_label, model in models:
        print(f"\n=== Evaluating {model_label} ===")
        metrics = run_evaluation(model, project_root / "output/05_normalized", split='test')
        print(metrics)

        comparison_rows.append({
            'model': model_label,
            'ADE': metrics.get('ADE'),
            'FDE': metrics.get('FDE'),
            'RMSE': metrics.get('RMSE'),
            'n_tracks': metrics.get('n_tracks'),
        })

        per_file_metrics = metrics.get('per_file_metrics', pd.DataFrame())
        per_month_metrics = metrics.get('per_month_metrics', pd.DataFrame())

        if not per_file_metrics.empty:
            per_file_path = diagnostics_dir / f"test_metrics_per_file_{model_key}.csv"
            per_file_metrics.to_csv(per_file_path, index=False)
            print("\nWorst 10 files by RMSE:")
            print(per_file_metrics.head(10).to_string(index=False))
            print(f"Saved per-file metrics to {per_file_path}")

        if not per_month_metrics.empty:
            per_month_path = diagnostics_dir / f"test_metrics_per_month_{model_key}.csv"
            per_month_metrics.to_csv(per_month_path, index=False)
            print("\nMonths sorted by RMSE:")
            print(per_month_metrics.to_string(index=False))
            print(f"Saved per-month metrics to {per_month_path}")

        # Plot ADE(t) for model on shared axis
        if fig is None or ax is None:
            fig, ax = plot_horizon_error(metrics, label=model_label)
        else:
            plot_horizon_error(metrics, label=model_label, ax=ax)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = diagnostics_dir / "test_metrics_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n=== Model comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"Saved model comparison to {comparison_path}")

    if ax is not None:
        ax.legend()
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