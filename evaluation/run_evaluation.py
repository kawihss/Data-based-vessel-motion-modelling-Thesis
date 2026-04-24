import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, HybridCVCTRVModel
from evaluation.evaluator import export_predictions_for_file, _resolve_files, _read_track_file, _iter_track_groups, reconstruct_positions, _extract_month_label
from evaluation.metrics import evaluate_trajectory

#runner for evaluation *testing, not tuning
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
    cv_row = tuned_values.get("cv") or tuned_values.get("constant_velocity")
    ctrv_row = tuned_values["ctrv"]
    hybrid_row = tuned_values["hybrid_cv_ctrv"]

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
        models.append(("constant_velocity", "Constant Velocity", ConstantVelocityModel(**cv_kwargs))) #key, label, instance
    if RUN_CTRV:
        models.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel(**ctrv_kwargs)))
    if RUN_HYBRID:
        models.append(("hybrid_cv_ctrv", "Hybrid CV/CTRV", HybridCVCTRVModel(**hybrid_kwargs)))

    if not models:
        raise SystemExit(0)

    comparison_rows = []
    data_dir = project_root / "output/07_parquet"
    files = _resolve_files(data_dir, EVAL_SPLIT, CONTEXT_FILTER)

    # Load all parquet files into memory ONCE
    file_data = [(f, _read_track_file(f)) for f in files]

    # Evaluate all models using the cached dataframes
    for model_key, model_label, model in models:
        per_file_rows = []
        all_true = []
        all_pred = []

        for file_path, df in file_data:
            file_true = []
            file_pred = []

            for track_id, context_df, pred_df in _iter_track_groups(df, with_track_id=True):
                n_pred_steps = len(pred_df)
                last_x = context_df['x'].iloc[-1]
                last_y = context_df['y'].iloc[-1]

                displacements = model.predict(context_df, n_pred_steps)
                pred_positions = reconstruct_positions(last_x, last_y, displacements)
                true_positions = pred_df[['x', 'y']].values

                if len(true_positions) != n_pred_steps:
                    continue

                file_true.append(true_positions)
                file_pred.append(pred_positions)

            if file_true:
                file_metrics = evaluate_trajectory(np.array(file_true), np.array(file_pred))
                per_file_rows.append({
                    'file': Path(file_path).name,
                    'month': _extract_month_label(file_path),
                    'ADE': file_metrics['ADE'],
                    'FDE': file_metrics['FDE'],
                    'RMSE': file_metrics['RMSE'],
                    'n_tracks': len(file_true),
                })
                all_true.extend(file_true)
                all_pred.extend(file_pred)

            if EXPORT_PREDICTIONS:
                export_predictions_for_file(
                    model, file_path, model_output_dir,
                    split=EVAL_SPLIT, model_key=model_key, model_label=model_label,
                )

        if all_true:
            metrics = evaluate_trajectory(np.array(all_true), np.array(all_pred))
            metrics['n_tracks'] = len(all_true)
        else:
            metrics = {
                'ADE': np.nan,
                'FDE': np.nan,
                'RMSE': np.nan,
                'ADE_per_step': np.array([]),
                'n_tracks': 0,
            }

        comparison_rows.append({
            'model': model_label,
            'ADE': metrics.get('ADE'),
            'FDE': metrics.get('FDE'),
            'RMSE': metrics.get('RMSE'),
            'n_tracks': metrics.get('n_tracks'),
        })

        per_file_metrics = pd.DataFrame(per_file_rows)
        if not per_file_metrics.empty:
            per_file_metrics = per_file_metrics.sort_values(['RMSE', 'FDE', 'ADE'], ascending=False).reset_index(drop=True)
            per_file_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_file_{model_key}.csv"
            per_file_metrics.to_csv(per_file_path, index=False)

            if per_file_metrics['month'].notna().any(): # only compute per-month metrics if month information is available
                per_month_metrics = (
                    per_file_metrics
                    .groupby('month', as_index=False)
                    .agg(
                        ADE=('ADE', 'mean'),
                        FDE=('FDE', 'mean'),
                        RMSE=('RMSE', 'mean'),
                        n_tracks=('n_tracks', 'sum'),
                        n_files=('file', 'count'),
                    )
                    .sort_values('RMSE', ascending=False)
                    .reset_index(drop=True)
                )
                per_month_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{model_key}.csv"
                per_month_metrics.to_csv(per_month_path, index=False)

        ade_per_step = metrics.get('ADE_per_step')
        if ade_per_step is not None and len(ade_per_step) > 0:
            ade_path = test_diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{model_key}.csv"
            pd.DataFrame({'ade': ade_per_step}).to_csv(ade_path, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(comparison_df.to_string(index=False))