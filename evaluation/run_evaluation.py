import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, HybridCVCTRVModel
from evaluation.evaluator import run_evaluation

#runner for evaluation. to be replaced by optuna framework
# Simple global runtime switches
RUN_CONSTANT_VELOCITY = True
RUN_CTRV = True
RUN_HYBRID = True
EVAL_SPLIT = 'test'
CONTEXT_FILTER = None  # e.g. 'lock' or ['harbour', 'lock']
EXPORT_PREDICTIONS = True


def _load_tuned_values(diagnostics_dir):
    df = pd.read_csv(diagnostics_dir / "tuning_best_params_val.csv")
    return {
        str(row.get("model_key", "")).strip().lower(): row.to_dict()
        for _, row in df.iterrows()
        if str(row.get("model_key", "")).strip()
    }

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    baseline_output_dir = project_root / "output" / "08_baseline_results"
    tuning_diagnostics_dir = project_root / "evaluation" / "diagnostics"  # where tuning summary is saved
    test_diagnostics_dir = baseline_output_dir / "diagnostics"  # where test results are saved
    model_output_dir = baseline_output_dir / "model_output"
    test_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tuned_values = _load_tuned_values(tuning_diagnostics_dir)
    cv_row = tuned_values.get("cv") or tuned_values.get("constant_velocity") or {}
    ctrv_row = tuned_values.get("ctrv") or {}
    hybrid_row = tuned_values.get("hybrid_cv_ctrv") or {}

    cv_steps = cv_row.get("best_velocity_steps")
    ctrv_steps = ctrv_row.get("best_velocity_steps")
    hybrid_cv_steps = hybrid_row.get("cv_velocity_steps")
    hybrid_ctrv_steps = hybrid_row.get("ctrv_velocity_steps")
    hybrid_rot_steps = hybrid_row.get("best_rot_steps")
    hybrid_rot_threshold = hybrid_row.get("best_rot_threshold")

    cv_kwargs = {"velocity_steps": int(cv_steps)}
    ctrv_kwargs = {"velocity_steps": int(ctrv_steps)}
    hybrid_kwargs = {
        "cv_velocity_steps": int(hybrid_cv_steps),
        "ctrv_velocity_steps": int(hybrid_ctrv_steps),
        "rot_steps": int(hybrid_rot_steps),
        "rot_threshold": float(hybrid_rot_threshold),
    }

    models = []
    if RUN_CONSTANT_VELOCITY:
        models.append(("constant_velocity", "Constant Velocity", ConstantVelocityModel(**cv_kwargs)))
    if RUN_CTRV:
        models.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel(**ctrv_kwargs)))
    if RUN_HYBRID:
        models.append(("hybrid_cv_ctrv", "Hybrid CV/CTRV", HybridCVCTRVModel(**hybrid_kwargs)))

    if not models:
        raise SystemExit(0)

    comparison_rows = []

    for model_key, model_label, model in models:
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
            per_file_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_file_{model_key}.csv"
            per_file_metrics.to_csv(per_file_path, index=False)

        if not per_month_metrics.empty:
            per_month_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{model_key}.csv"
            per_month_metrics.to_csv(per_month_path, index=False)

        ade_per_step = metrics.get('ADE_per_step')
        if ade_per_step is not None and len(ade_per_step) > 0:
            ade_path = test_diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{model_key}.csv"
            pd.DataFrame({'ade': ade_per_step}).to_csv(ade_path, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)