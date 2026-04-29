import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, ConstantTurnRateVelocityArcModel, HybridCVCTRVModel
from models.filters import KalmanFilter, CTRVExtendedKalmanFilter
from evaluation.evaluator import export_predictions_for_file, _resolve_files, _read_track_file, _iter_track_groups, reconstruct_positions, _extract_month_label
from evaluation.metrics import evaluate_trajectory
from evaluation.runtime_config import load_runtime_config, resolve_run_paths, update_latest_run_pointer, write_run_metadata, get_sampling_value, subsample_items

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_runtime_config(PROJECT_ROOT)

# runner for evaluation *testing, not tuning
RUN_CONSTANT_VELOCITY = bool(CONFIG["models"]["cv"])
RUN_CTRV = bool(CONFIG["models"]["ctrv"])
RUN_CTRV_ARC = bool(CONFIG["models"]["ctrv_arc"])
RUN_HYBRID = bool(CONFIG["models"]["hybrid"])
RUN_KALMAN = bool(CONFIG["models"]["kalman"])
RUN_CTRV_EKF = bool(CONFIG["models"]["ctrv_ekf"])
EVAL_SPLIT = str(CONFIG["data"]["split"])
CONTEXT_FILTER_EVALUATION = CONFIG["data"]["context_filter_evaluation"]
EXPORT_PREDICTIONS = bool(CONFIG["evaluation"]["export_predictions"])
SEED = int(CONFIG["run"]["seed"])
SAMPLE_PCT = int(CONFIG["run"]["sample_pct"])
EVALUATION_PCT = int(get_sampling_value(CONFIG, "evaluation_pct"))

np.random.seed(SEED)


def _load_tuned_values(diagnostics_dir):
    df = pd.read_csv(diagnostics_dir / "tuning_best_params_val.csv")
    return {
        str(row.get("model_key", "")).strip().lower(): row.to_dict()
        for _, row in df.iterrows()
        if str(row.get("model_key", "")).strip()
    }

if __name__ == "__main__":
    run_paths = resolve_run_paths(PROJECT_ROOT, CONFIG, create=True)
    tuning_diagnostics_dir = run_paths["tuning_dir"]
    test_diagnostics_dir = run_paths["diagnostics_dir"]
    model_output_dir = run_paths["model_output_dir"]

    comparison_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_model_comparison.csv"
    if comparison_path.exists():
        print(f"[Warning] Existing evaluation outputs found in {test_diagnostics_dir}. Files will be overwritten for run '{CONFIG['run']['name']}'.")

    print(
        f"Run: {CONFIG['run']['name']} | split={EVAL_SPLIT} | seed={SEED} "
        f"| sample_pct={SAMPLE_PCT}% | evaluation_pct={EVALUATION_PCT}%"
    )

    tuned_values = _load_tuned_values(tuning_diagnostics_dir)
    cv_row = tuned_values.get("cv") or tuned_values.get("constant_velocity")
    ctrv_row = tuned_values.get("ctrv")
    ctrv_arc_row = tuned_values.get("ctrv_arc")
    hybrid_row = tuned_values.get("hybrid_cv_ctrv")
    kalman_row = tuned_values.get("kalman")
    ctrv_ekf_row = tuned_values.get("ctrv_ekf")

    cv_kwargs = {}
    if cv_row is not None and pd.notna(cv_row.get("best_velocity_steps")):
        cv_kwargs = {"velocity_steps": int(cv_row.get("best_velocity_steps"))}

    ctrv_kwargs = {}
    if ctrv_row is not None and pd.notna(ctrv_row.get("best_velocity_steps")):
        ctrv_kwargs = {"velocity_steps": int(ctrv_row.get("best_velocity_steps"))}

    ctrv_arc_kwargs = {}
    if ctrv_arc_row is not None and pd.notna(ctrv_arc_row.get("best_velocity_steps")):
        ctrv_arc_kwargs = {"velocity_steps": int(ctrv_arc_row.get("best_velocity_steps"))}

    hybrid_kwargs = {}
    if hybrid_row is not None:
        if all(pd.notna(hybrid_row.get(k)) for k in ["cv_velocity_steps", "ctrv_velocity_steps", "best_rot_steps", "best_rot_threshold"]):
            hybrid_kwargs = {
                "cv_velocity_steps": int(hybrid_row.get("cv_velocity_steps")),
                "ctrv_velocity_steps": int(hybrid_row.get("ctrv_velocity_steps")),
                "rot_steps": int(hybrid_row.get("best_rot_steps")),
                "rot_threshold": float(hybrid_row.get("best_rot_threshold")),
            }

    kalman_kwargs = {}
    if kalman_row is not None:
        required = ["best_q_pos", "best_q_vel", "best_r_pos", "best_p0_pos", "best_p0_vel"]
        if all(pd.notna(kalman_row.get(k)) for k in required):
            kalman_kwargs = {
                "q_pos": float(kalman_row.get("best_q_pos")),
                "q_vel": float(kalman_row.get("best_q_vel")),
                "r_pos": float(kalman_row.get("best_r_pos")),
                "p0_pos": float(kalman_row.get("best_p0_pos")),
                "p0_vel": float(kalman_row.get("best_p0_vel")),
            }

    ctrv_ekf_kwargs = {}
    if ctrv_ekf_row is not None:
        required = ["best_q_pos", "best_q_vel", "best_q_rot", "best_r_pos", "best_p0_pos", "best_p0_vel", "best_p0_rot", "best_init_velocity_steps"]
        if all(pd.notna(ctrv_ekf_row.get(k)) for k in required):
            ctrv_ekf_kwargs = {
                "q_pos": float(ctrv_ekf_row.get("best_q_pos")),
                "q_vel": float(ctrv_ekf_row.get("best_q_vel")),
                "q_rot": float(ctrv_ekf_row.get("best_q_rot")),
                "r_pos": float(ctrv_ekf_row.get("best_r_pos")),
                "p0_pos": float(ctrv_ekf_row.get("best_p0_pos")),
                "p0_vel": float(ctrv_ekf_row.get("best_p0_vel")),
                "p0_rot": float(ctrv_ekf_row.get("best_p0_rot")),
                "init_velocity_steps": int(ctrv_ekf_row.get("best_init_velocity_steps")),
            }

    models = []
    if RUN_CONSTANT_VELOCITY:
        if not cv_kwargs:
            raise ValueError("CV is enabled but no tuned CV parameters were found in tuning_best_params_val.csv")
        models.append(("constant_velocity", "Constant Velocity", ConstantVelocityModel(**cv_kwargs))) #key, label, instance
    if RUN_CTRV:
        if not ctrv_kwargs:
            raise ValueError("CTRV is enabled but no tuned CTRV parameters were found in tuning_best_params_val.csv")
        models.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel(**ctrv_kwargs)))
    if RUN_CTRV_ARC:
        if not ctrv_arc_kwargs:
            raise ValueError("CTRV Arc is enabled but no tuned CTRV Arc parameters were found in tuning_best_params_val.csv")
        models.append(("ctrv_arc", "CTRV Arc", ConstantTurnRateVelocityArcModel(**ctrv_arc_kwargs)))
    if RUN_HYBRID:
        if not hybrid_kwargs:
            raise ValueError("Hybrid is enabled but no tuned Hybrid parameters were found in tuning_best_params_val.csv")
        models.append(("hybrid_cv_ctrv", "Hybrid CV/CTRV", HybridCVCTRVModel(**hybrid_kwargs)))
    if RUN_KALMAN:
        if not kalman_kwargs:
            raise ValueError("Kalman is enabled but no tuned Kalman parameters were found in tuning_best_params_val.csv")
        models.append(("kalman", "Kalman", KalmanFilter(**kalman_kwargs)))
    if RUN_CTRV_EKF:
        if not ctrv_ekf_kwargs:
            raise ValueError("CTRV EKF is enabled but no tuned CTRV EKF parameters were found in tuning_best_params_val.csv")
        models.append(("ctrv_ekf", "CTRV EKF", CTRVExtendedKalmanFilter(**ctrv_ekf_kwargs)))

    if not models:
        raise SystemExit(0)

    comparison_rows = []
    data_dir = PROJECT_ROOT / CONFIG["data"]["parquet_dir"]
    files = _resolve_files(data_dir, EVAL_SPLIT, CONTEXT_FILTER_EVALUATION)
    files = subsample_items(files, sample_pct=EVALUATION_PCT, seed=SEED)
    total_models = len(models)
    total_files = len(files)

    print(
        f"Evaluating {total_models} model(s) on {total_files} file(s) "
        f"for split '{EVAL_SPLIT}'"
        + (f", context(s) {CONTEXT_FILTER_EVALUATION}" if CONTEXT_FILTER_EVALUATION else "")
    )

    # Load all parquet files into memory ONCE
    file_data = [(f, _read_track_file(f)) for f in files]

    # Evaluate all models using the cached dataframes
    for model_index, (model_key, model_label, model) in enumerate(models, start=1):
        per_file_rows = []
        all_true = []
        all_pred = []
        completed_files = 0

        print(f"[Progress] model {model_index}/{total_models} started ({model_label})")

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
                    'sample_pct': EVALUATION_PCT,
                })
                all_true.extend(file_true)
                all_pred.extend(file_pred)

            if EXPORT_PREDICTIONS:
                export_predictions_for_file(
                    model, file_path, model_output_dir,
                    split=EVAL_SPLIT, model_key=model_key, model_label=model_label,
                )

            completed_files += 1
            print(
                f"[Progress] model {model_index}/{total_models}, "
                f"file {completed_files}/{total_files} completed "
                f"({model_label}, {Path(file_path).name})"
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
            'sample_pct': EVALUATION_PCT,
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
                per_month_metrics['sample_pct'] = EVALUATION_PCT
                per_month_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{model_key}.csv"
                per_month_metrics.to_csv(per_month_path, index=False)

        ade_per_step = metrics.get('ADE_per_step')
        if ade_per_step is not None and len(ade_per_step) > 0:
            ade_path = test_diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{model_key}.csv"
            pd.DataFrame({'ade': ade_per_step}).to_csv(ade_path, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_path, index=False)

    write_run_metadata(run_paths, CONFIG, stage="evaluation")
    update_latest_run_pointer(run_paths)

    print(comparison_df.to_string(index=False))