import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, ConstantTurnRateVelocityArcModel, HybridCVCTRVModel
from models.filters import KalmanFilter, CTRVExtendedKalmanFilter
from models.sequence import TirexLSTMModel, Chronos2ZeroShotModel
from evaluation.evaluator import export_predictions_for_file, _resolve_files, _read_track_file, _iter_track_groups, reconstruct_positions, _extract_month_label
from evaluation.metrics import evaluate_trajectory, evaluate_quantile_forecast, calculate_channel_importance, calculate_timestep_importance
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
RUN_TIREX_LSTM = bool(CONFIG["models"]["tirex_lstm"])
RUN_CHRONOS2_ZERO_SHOT = bool(CONFIG["models"].get("chronos2_zero_shot", False))
TIREX_CFG = CONFIG["models"].get("tirex", {})
TIREX_MODEL_NAME = str(TIREX_CFG.get("model_name", "NX-AI/TiRex"))
TIREX_DEVICE = TIREX_CFG.get("device", None) # read
TIREX_DEVICE = None if TIREX_DEVICE is None else (str(TIREX_DEVICE).strip() or None) # convert
TIREX_BACKEND = str(TIREX_CFG.get("backend", "torch"))
TIREX_COMPILE_MODEL = bool(TIREX_CFG.get("compile_model", False))
TIREX_SCALER_PATH = TIREX_CFG.get("scaler_path", "output/05_normalized/scalers.pkl") # read
TIREX_SCALER_PATH = str(TIREX_SCALER_PATH).strip() if TIREX_SCALER_PATH is not None else "output/05_normalized/scalers.pkl" # convert
CHRONOS2_CFG = CONFIG["models"].get("chronos2", {})
CHRONOS2_MODEL_NAME = str(CHRONOS2_CFG.get("model_name", "amazon/chronos-2"))
CHRONOS2_DEVICE_MAP = CHRONOS2_CFG.get("device_map", None)
CHRONOS2_DEVICE_MAP = None if CHRONOS2_DEVICE_MAP is None else (str(CHRONOS2_DEVICE_MAP).strip() or None)
CHRONOS2_MAX_MEMORY = CHRONOS2_CFG.get("max_memory", None)
CHRONOS2_TORCH_DTYPE = CHRONOS2_CFG.get("torch_dtype", None)
CHRONOS2_TORCH_DTYPE = None if CHRONOS2_TORCH_DTYPE is None else (str(CHRONOS2_TORCH_DTYPE).strip() or None)
CHRONOS2_SCALER_PATH = CHRONOS2_CFG.get("scaler_path", "output/05_normalized/scalers.pkl")
CHRONOS2_SCALER_PATH = str(CHRONOS2_SCALER_PATH).strip() if CHRONOS2_SCALER_PATH is not None else "output/05_normalized/scalers.pkl"
EVAL_SPLIT = str(CONFIG["data"]["split"])
CONTEXT_FILTER_EVALUATION = CONFIG["data"]["context_filter_evaluation"]
EXPORT_PREDICTIONS = bool(CONFIG["evaluation"]["export_predictions"])
SEED = int(CONFIG["run"]["seed"])
SAMPLE_PCT = int(CONFIG["run"]["sample_pct"])
EVALUATION_PCT = int(get_sampling_value(CONFIG, "evaluation_pct"))

np.random.seed(SEED)
print(EVAL_SPLIT)
CHRONOS_COVARIATE_COLUMNS = (
    "dx_norm",
    "dy_norm",
    "sog_norm",
    "cog_sin_norm",
    "cog_cos_norm",
    "dt_norm",
    "rot_norm",
)
REPORT_CONTEXTS = ("harbour", "river", "channel", "lock")

_DXDY_SCALER_PARAMS_CACHE = {}

def _load_dxdy_scaler_params(scaler_path_str):
    """Load dx/dy mean and scale from scalers.pkl, cached by path."""
    cached = _DXDY_SCALER_PARAMS_CACHE.get(scaler_path_str)
    if cached is not None:
        return cached
    scalers = joblib.load(scaler_path_str)
    global_scaler = scalers.get("global_scaler", {})
    means = np.asarray(global_scaler.get("mean", []), dtype=float)
    scales = np.asarray(global_scaler.get("scale", []), dtype=float)
    params = {
        "dx_mean": float(means[0]),
        "dx_scale": float(scales[0]),
        "dy_mean": float(means[1]),
        "dy_scale": float(scales[1]),
    }
    _DXDY_SCALER_PARAMS_CACHE[scaler_path_str] = params
    return params


def _denormalize_true_displacements(dx_norm, dy_norm, scaler_path_str):
    """Convert normalized dx_norm/dy_norm ground-truth to real metres."""
    p = _load_dxdy_scaler_params(scaler_path_str)
    dx = p["dx_scale"] * np.asarray(dx_norm, dtype=float) + p["dx_mean"]
    dy = p["dy_scale"] * np.asarray(dy_norm, dtype=float) + p["dy_mean"]
    return np.column_stack([dx, dy])


def _load_tuned_values(diagnostics_dir):
    df = pd.read_csv(diagnostics_dir / "tuning_best_params_val.csv")
    return {
        str(row.get("model_key", "")).strip().lower(): row.to_dict()
        for _, row in df.iterrows()
        if str(row.get("model_key", "")).strip()
    }


def _requires_tuned_values():
    return any([
        RUN_CONSTANT_VELOCITY,
        RUN_CTRV,
        RUN_CTRV_ARC,
        RUN_HYBRID,
        RUN_KALMAN,
        RUN_CTRV_EKF,
        RUN_TIREX_LSTM,
    ])


def _stack_quantile_lists(quantile_predictions):
    return {
        float(quantile): np.asarray(predictions, dtype=float)
        for quantile, predictions in quantile_predictions.items()
    }


def _empty_overall_metrics():
    return {
        'ADE': np.nan,
        'FDE': np.nan,
        'RMSE': np.nan,
        'ADE_per_step': np.array([]),
        'n_tracks': 0,
    }


def _make_context_bucket():
    return {
        'true_pos': [],
        'pred_pos': [],
        'true_disp': [],
        'quantiles': {},
        'covariates': {column: [] for column in CHRONOS_COVARIATE_COLUMNS},
        'motion_magnitude': [],
        'future_motion_score': [],
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

    tuned_values = _load_tuned_values(tuning_diagnostics_dir) if _requires_tuned_values() else {}
    cv_row = tuned_values.get("cv") or tuned_values.get("constant_velocity")
    ctrv_row = tuned_values.get("ctrv")
    ctrv_arc_row = tuned_values.get("ctrv_arc")
    hybrid_row = tuned_values.get("hybrid_cv_ctrv")
    kalman_row = tuned_values.get("kalman")
    ctrv_ekf_row = tuned_values.get("ctrv_ekf")
    tirex_row = tuned_values.get("tirex_lstm")

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

    tirex_kwargs = {}
    if tirex_row is not None and pd.notna(tirex_row.get("best_velocity_steps")):
        tirex_kwargs = {
            "velocity_steps": int(tirex_row.get("best_velocity_steps")),
            "model_name": TIREX_MODEL_NAME,
            "device": TIREX_DEVICE,
            "backend": TIREX_BACKEND,
            "compile_model": TIREX_COMPILE_MODEL,
            "scaler_path": str(PROJECT_ROOT / TIREX_SCALER_PATH),
        }

    chronos2_kwargs = {}
    if RUN_CHRONOS2_ZERO_SHOT:
        chronos2_kwargs = {
            "model_name": CHRONOS2_MODEL_NAME,
            "device_map": CHRONOS2_DEVICE_MAP,
            "max_memory": CHRONOS2_MAX_MEMORY,
            "torch_dtype": CHRONOS2_TORCH_DTYPE,
            "scaler_path": str(PROJECT_ROOT / CHRONOS2_SCALER_PATH),
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
    if RUN_TIREX_LSTM:
        if not tirex_kwargs:
            raise ValueError("TiRex LSTM is enabled but no tuned TiRex LSTM parameters were found in tuning_best_params_val.csv")
        models.append(("tirex_lstm", "TiRex LSTM", TirexLSTMModel(**tirex_kwargs)))
    if RUN_CHRONOS2_ZERO_SHOT:
        models.append(("chronos2_zero_shot", "Chronos-2 Zero-Shot", Chronos2ZeroShotModel(**chronos2_kwargs)))

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
        all_true_disp = []
        all_quantile_predictions = {}
        per_context_data = {context: _make_context_bucket() for context in REPORT_CONTEXTS}
        completed_files = 0

        print(f"[Progress] model {model_index}/{total_models} started ({model_label})")

        for file_path, df in file_data:
            file_true = []
            file_pred = []
            file_true_disp = []
            file_quantile_predictions = {}

            for track_id, context_df, pred_df in _iter_track_groups(df, with_track_id=True):
                n_pred_steps = len(pred_df)
                last_x = context_df['x'].iloc[-1]
                last_y = context_df['y'].iloc[-1]

                quantile_predictions = model.predict_quantiles(context_df, n_pred_steps) if hasattr(model, 'predict_quantiles') else None
                displacements = np.asarray(quantile_predictions[0.5], dtype=float) if quantile_predictions is not None else model.predict(context_df, n_pred_steps)
                pred_positions = reconstruct_positions(last_x, last_y, displacements)
                true_positions = pred_df[['x', 'y']].values
                # Denormalize ground-truth displacements to real metres so they match
                # the denormalized quantile predictions from Chronos/TiRex.
                _active_scaler_path = str(PROJECT_ROOT / CHRONOS2_SCALER_PATH)
                true_displacements = _denormalize_true_displacements(
                    pred_df['dx_norm'].values, pred_df['dy_norm'].values, _active_scaler_path
                )
                context_label = str(context_df['context'].iloc[-1])
                ctx_sorted = context_df.sort_values('t_utc')

                if len(true_positions) != n_pred_steps:
                    continue

                file_true.append(true_positions)
                file_pred.append(pred_positions)
                if quantile_predictions is not None:
                    file_true_disp.append(true_displacements)
                    for quantile, prediction in quantile_predictions.items():
                        file_quantile_predictions.setdefault(float(quantile), []).append(np.asarray(prediction, dtype=float))

                if context_label in per_context_data:
                    bucket = per_context_data[context_label]
                    bucket['true_pos'].append(true_positions)
                    bucket['pred_pos'].append(pred_positions)
                    if quantile_predictions is not None:
                        bucket['true_disp'].append(true_displacements)
                        for quantile, prediction in quantile_predictions.items():
                            bucket['quantiles'].setdefault(float(quantile), []).append(np.asarray(prediction, dtype=float))
                        for covariate in CHRONOS_COVARIATE_COLUMNS:
                            bucket['covariates'][covariate].append(ctx_sorted[covariate].to_numpy(dtype=float, copy=True))
                        bucket['motion_magnitude'].append(
                            np.abs(ctx_sorted['dx_norm'].to_numpy(dtype=float, copy=True)) + np.abs(ctx_sorted['dy_norm'].to_numpy(dtype=float, copy=True))
                        )
                        bucket['future_motion_score'].append(
                            float(np.sqrt((true_displacements ** 2).sum(axis=1)).mean())
                        )

            if file_true:
                file_metrics = evaluate_trajectory(np.array(file_true), np.array(file_pred))
                file_row = {
                    'file': Path(file_path).name,
                    'month': _extract_month_label(file_path),
                    'ADE': file_metrics['ADE'],
                    'FDE': file_metrics['FDE'],
                    'RMSE': file_metrics['RMSE'],
                    'n_tracks': len(file_true),
                    'sample_pct': EVALUATION_PCT,
                }
                if file_true_disp:
                    file_row.update(
                        evaluate_quantile_forecast(
                            np.asarray(file_true_disp, dtype=float),
                            _stack_quantile_lists(file_quantile_predictions),
                        )
                    )
                per_file_rows.append(file_row)
                all_true.extend(file_true)
                all_pred.extend(file_pred)
                if file_true_disp:
                    all_true_disp.extend(file_true_disp)
                    for quantile, prediction in file_quantile_predictions.items():
                        all_quantile_predictions.setdefault(float(quantile), []).extend(prediction)

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
            if all_true_disp:
                metrics.update(
                    evaluate_quantile_forecast(
                        np.asarray(all_true_disp, dtype=float),
                        _stack_quantile_lists(all_quantile_predictions),
                    )
                )
        else:
            metrics = _empty_overall_metrics()

        comparison_rows.append({
            'model': model_label,
            'ADE': metrics.get('ADE'),
            'FDE': metrics.get('FDE'),
            'RMSE': metrics.get('RMSE'),
            'MIW': metrics.get('MIW', np.nan),
            'Coverage': metrics.get('Coverage', np.nan),
            'IQR': metrics.get('IQR', np.nan),
            'Winkler80': metrics.get('Winkler80', np.nan),
            'n_tracks': metrics.get('n_tracks'),
            'sample_pct': EVALUATION_PCT,
        })

        per_file_metrics = pd.DataFrame(per_file_rows)
        if not per_file_metrics.empty:
            sort_columns = [column for column in ['RMSE', 'FDE', 'ADE'] if column in per_file_metrics.columns]
            per_file_metrics = per_file_metrics.sort_values(sort_columns, ascending=False).reset_index(drop=True)
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

        per_context_rows = []
        channel_importance_rows = []
        timestep_importance_rows = []
        for context_label, bucket in per_context_data.items():
            if not bucket['true_pos']:
                continue

            context_metrics = evaluate_trajectory(
                np.asarray(bucket['true_pos'], dtype=float),
                np.asarray(bucket['pred_pos'], dtype=float),
            )
            context_row = {
                'context': context_label,
                'ADE': context_metrics['ADE'],
                'FDE': context_metrics['FDE'],
                'RMSE': context_metrics['RMSE'],
                'n_tracks': len(bucket['true_pos']),
                'sample_pct': EVALUATION_PCT,
            }
            if bucket['true_disp']:
                context_row.update(
                    evaluate_quantile_forecast(
                        np.asarray(bucket['true_disp'], dtype=float),
                        _stack_quantile_lists(bucket['quantiles']),
                    )
                )
                importance = calculate_channel_importance(bucket['covariates'], bucket['motion_magnitude'])
                for covariate, value in importance.items():
                    channel_importance_rows.append({
                        'context': context_label,
                        'covariate': covariate,
                        'importance': value,
                        'sample_pct': EVALUATION_PCT,
                    })
                timestep_importance = calculate_timestep_importance(
                    bucket['motion_magnitude'],
                    bucket['future_motion_score'],
                )
                for timestep, value in timestep_importance.items():
                    timestep_importance_rows.append({
                        'context': context_label,
                        'timestep': int(timestep),
                        'importance': value,
                        'sample_pct': EVALUATION_PCT,
                    })
            per_context_rows.append(context_row)

        if per_context_rows:
            per_context_path = test_diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_context_{model_key}.csv"
            pd.DataFrame(per_context_rows).to_csv(per_context_path, index=False)
        if channel_importance_rows:
            importance_path = test_diagnostics_dir / f"{EVAL_SPLIT}_channel_importance_{model_key}.csv"
            pd.DataFrame(channel_importance_rows).to_csv(importance_path, index=False)
        if timestep_importance_rows:
            timestep_path = test_diagnostics_dir / f"{EVAL_SPLIT}_timestep_importance_{model_key}.csv"
            pd.DataFrame(timestep_importance_rows).to_csv(timestep_path, index=False)

        ade_per_step = metrics.get('ADE_per_step')
        if ade_per_step is not None and len(ade_per_step) > 0:
            ade_path = test_diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{model_key}.csv"
            pd.DataFrame({'ade': ade_per_step}).to_csv(ade_path, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_path, index=False)

    write_run_metadata(run_paths, CONFIG, stage="evaluation")
    update_latest_run_pointer(run_paths)

    print(comparison_df.to_string(index=False))