import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel
from evaluation.evaluator import run_evaluation, plot_horizon_error
import matplotlib.pyplot as plt

#runner for evaluation. to be replaced by optuna framework
# Simple global runtime switches
RUN_CONSTANT_VELOCITY = True
RUN_CTRV = True
EVAL_SPLIT = 'test'
CONTEXT_FILTER = None  # e.g. 'lock' or ['harbour', 'lock']
EXPORT_PREDICTIONS = True
SHOW_PLOT = True 


def _load_best_velocity_steps(diagnostics_dir):
    best_path = diagnostics_dir / "tuning_best_params_val.csv"
    if not best_path.exists():
        print(f"No tuning summary found at {best_path}; using fallback defaults.")
        return {}

    df = pd.read_csv(best_path)
    if df.empty:
        print(f"Tuning summary {best_path} is empty; using fallback defaults.")
        return {}

    if "model_key" not in df.columns:
        print("Tuning summary missing 'model_key' column; using fallback defaults.")
        return {}

    step_col = None
    for candidate in ("best_velocity_steps", "best_velocity_fraction"):
        if candidate in df.columns:
            step_col = candidate
            break

    if step_col is None:
        print("Tuning summary missing best parameter column; using fallback defaults.")
        return {}

    best_values = {}
    for _, row in df.iterrows():
        key = str(row.get("model_key", "")).strip().lower()
        value = row.get(step_col)
        if key and pd.notna(value):
            best_values[key] = value

    if best_values:
        print(f"Loaded tuned parameters from {best_path}")
    else:
        print(f"No valid tuned parameters in {best_path}; using fallback defaults.")

    return best_values

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    baseline_output_dir = project_root / "output" / "08_baseline_results"
    diagnostics_dir = baseline_output_dir / "diagnostics"
    model_output_dir = baseline_output_dir / "model_output"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tuned_values = _load_best_velocity_steps(diagnostics_dir)
    cv_steps = tuned_values.get("cv") or tuned_values.get("constant_velocity")
    ctrv_steps = tuned_values.get("ctrv")

    cv_kwargs = {"velocity_fraction": 0.4} if cv_steps is None else {"velocity_steps": int(cv_steps)}
    ctrv_kwargs = {"velocity_fraction": 0.4} if ctrv_steps is None else {"velocity_steps": int(ctrv_steps)}

    models = []
    if RUN_CONSTANT_VELOCITY:
        models.append(("constant_velocity", "Constant Velocity", ConstantVelocityModel(**cv_kwargs)))
    if RUN_CTRV:
        models.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel(**ctrv_kwargs)))

    if not models:
        print("No models selected. Set RUN_CONSTANT_VELOCITY and/or RUN_CTRV to True.")
        raise SystemExit(0)

    comparison_rows = []

    for model_key, model_label, model in models:
        print(f"\n=== Evaluating {model_label} ===")
        metrics = run_evaluation(
            model,
            project_root / "output/07_parquet",
            split=EVAL_SPLIT,
            context_filter=CONTEXT_FILTER,
            export_predictions=EXPORT_PREDICTIONS,
            prediction_output_dir=model_output_dir,
            model_key=model_key,
            model_label=model_label,
        )
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
            per_file_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_file_{model_key}.csv"
            per_file_metrics.to_csv(per_file_path, index=False)
            print("\nWorst 10 files by RMSE:")
            print(per_file_metrics.head(10).to_string(index=False))
            print(f"Saved per-file metrics to {per_file_path}")

        if not per_month_metrics.empty:
            per_month_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{model_key}.csv"
            per_month_metrics.to_csv(per_month_path, index=False)
            print("\nMonths sorted by RMSE:")
            print(per_month_metrics.to_string(index=False))
            print(f"Saved per-month metrics to {per_month_path}")

        prediction_exports = metrics.get('prediction_exports', pd.DataFrame())
        if not prediction_exports.empty:
            print(f"Saved {len(prediction_exports)} prediction file(s) for {model_label} to {model_output_dir}")
            print(prediction_exports.head(5).to_string(index=False))

        # Persist ADE_per_step so the plot block below can load it independently
        ade_per_step = metrics.get('ADE_per_step')
        if ade_per_step is not None and len(ade_per_step) > 0:
            ade_path = diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{model_key}.csv"
            pd.DataFrame({'ade': ade_per_step}).to_csv(ade_path, index=False)
            print(f"Saved ADE_per_step to {ade_path}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n=== Model comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"Saved model comparison to {comparison_path}")

    # Separate plot block: reads saved ADE_per_step for ALL known models,
    # regardless of which were selected for this run.
    if SHOW_PLOT:
        ALL_MODEL_LABELS = {
            "constant_velocity": "Constant Velocity",
            "ctrv": "CTRV",
        }
        fig_p, ax_p = plt.subplots()
        any_plotted = False
        for key, label in ALL_MODEL_LABELS.items():
            ade_path = diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{key}.csv"
            if not ade_path.exists():
                print(f"No ADE_per_step file for {label}, skipping.")
                continue
            ade_values = pd.read_csv(ade_path)['ade'].values
            plot_horizon_error({'ADE_per_step': ade_values}, label=label, ax=ax_p)
            any_plotted = True
        if any_plotted:
            ax_p.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No ADE_per_step files found for plotting.")