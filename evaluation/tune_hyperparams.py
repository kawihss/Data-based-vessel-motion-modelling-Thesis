import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["CUDA_LIB"] = "/usr/local/cuda-12.6/targets/x86_64-linux/lib"
os.environ["XLSTM_EXTRA_INCLUDE_PATHS"] = "/usr/local/cuda-12.6/include"

import pandas as pd
import numpy as np
import optuna
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, ConstantTurnRateVelocityArcModel, HybridCVCTRVModel
from models.filters import KalmanFilter, CTRVExtendedKalmanFilter
from models.sequence import TirexLSTMModel
from models.sequence.minimal_lstm import MinimalLSTMNet, CONTEXT_LEN, PRED_LEN, FEATURE_COLUMNS, TARGET_COLUMNS
from evaluation.evaluator import load_tracks_cached_numpy, evaluate_model_cached_numpy, _resolve_files, _read_track_file, _iter_track_groups
from evaluation.runtime_config import load_runtime_config, resolve_run_paths, update_latest_run_pointer, write_run_metadata, get_sampling_value

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_runtime_config(PROJECT_ROOT)

CONTEXT_FILTER_TUNING = CONFIG["data"]["context_filter_tuning"]
RUN_CV = bool(CONFIG["models"]["cv"])
RUN_CTRV = bool(CONFIG["models"]["ctrv"])
RUN_CTRV_ARC = bool(CONFIG["models"]["ctrv_arc"])
RUN_HYBRID = bool(CONFIG["models"]["hybrid"])
RUN_KALMAN = bool(CONFIG["models"]["kalman"])
RUN_CTRV_EKF = bool(CONFIG["models"]["ctrv_ekf"])
RUN_TIREX_LSTM = bool(CONFIG["models"]["tirex_lstm"])
RUN_MINIMAL_LSTM = bool(CONFIG["models"].get("minimal_lstm", False))
MINIMAL_LSTM_CFG = CONFIG["models"].get("minimal_lstm_cfg", {})
MINIMAL_LSTM_DEVICE = str(MINIMAL_LSTM_CFG.get("device", "cpu"))
MINIMAL_LSTM_CHECKPOINT = str(MINIMAL_LSTM_CFG.get("checkpoint_path", ""))
MINIMAL_LSTM_MAX_EPOCHS = int(MINIMAL_LSTM_CFG.get("max_epochs", 60))
MINIMAL_LSTM_PATIENCE = int(MINIMAL_LSTM_CFG.get("patience", 8))
MINIMAL_LSTM_MIN_DELTA = float(MINIMAL_LSTM_CFG.get("min_delta", 1e-4))
MINIMAL_LSTM_TUNING_CFG = CONFIG["tuning"].get("minimal_lstm", {})
MINIMAL_LSTM_N_TRIALS = int(CONFIG["tuning"].get("n_trials_minimal_lstm", 30))
MINIMAL_LSTM_PRELOAD_TO_GPU = bool(MINIMAL_LSTM_CFG.get("preload_to_gpu", False))
TIREX_CFG = CONFIG["models"].get("tirex", {})
TIREX_MODEL_NAME = str(TIREX_CFG.get("model_name", "NX-AI/TiRex"))
TIREX_DEVICE = TIREX_CFG.get("device", None)
TIREX_DEVICE = None if TIREX_DEVICE is None else (str(TIREX_DEVICE).strip() or None)
TIREX_BACKEND = str(TIREX_CFG.get("backend", "torch"))
TIREX_COMPILE_MODEL = bool(TIREX_CFG.get("compile_model", False))
TIREX_BATCH_SIZE = int(TIREX_CFG.get("batch_size", 1))
TIREX_SCALER_PATH = TIREX_CFG.get("scaler_path", "output/05_normalized/scalers.pkl")
TIREX_SCALER_PATH = str(TIREX_SCALER_PATH).strip() if TIREX_SCALER_PATH is not None else "output/05_normalized/scalers.pkl"
HYBRID_N_TRIALS = int(CONFIG["tuning"]["n_trials_hybrid"])
KALMAN_N_TRIALS = int(CONFIG["tuning"]["n_trials_kalman"])
CTRV_EKF_N_TRIALS = int(CONFIG["tuning"]["n_trials_ctrv_ekf"])
EARLY_STOPPING_PATIENCE = int(CONFIG["tuning"]["early_stopping_patience"])
EARLY_STOPPING_MIN_DELTA = float(CONFIG["tuning"]["early_stopping_min_delta"])
SEED = int(CONFIG["run"]["seed"])
SAMPLE_PCT = int(CONFIG["run"]["sample_pct"])
TUNING_VALIDATION_PCT = int(get_sampling_value(CONFIG, "tuning_validation_pct"))

np.random.seed(SEED)


def _build_lstm_samples(split, sample_pct=100, seed=SEED):
    data_dir = PROJECT_ROOT / CONFIG["data"]["parquet_dir"]
    context_filter = CONFIG["data"].get("context_filter_tuning")
    files = _resolve_files(data_dir, split, context_filter)
    if sample_pct < 100:
        rng = np.random.default_rng(seed)
        n = max(1, int(len(files) * sample_pct / 100))
        files = rng.choice(files, size=n, replace=False).tolist()
    x_list, y_list = [], []
    for file_path in files:
        df = _read_track_file(file_path)
        for context_df, pred_df in _iter_track_groups(df):
            if len(context_df) < CONTEXT_LEN or len(pred_df) < PRED_LEN:
                continue
            x_list.append(context_df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-CONTEXT_LEN:])
            y_list.append(pred_df.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)[:PRED_LEN])
    if not x_list:
        raise RuntimeError(f"No samples for split '{split}'")
    return np.stack(x_list), np.stack(y_list)


def _make_lstm_tensors(x_train, y_train, x_val, y_val, device, preload_to_gpu=False):
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    if preload_to_gpu:
        return x_train_t.to(device), y_train_t.to(device), x_val_t.to(device), y_val_t.to(device), False, True

    pin_memory = device.type == "cuda"
    return x_train_t, y_train_t, x_val_t, y_val_t, pin_memory, False


def _lstm_eval_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x_b, y_b in loader:
            total += criterion(model(x_b.to(device)), y_b.to(device)).item() * len(x_b)
            count += len(x_b)
    return total / count


def _run_lstm_training(
    params,
    x_train,
    y_train,
    x_val,
    y_val,
    device_str,
    max_epochs,
    patience,
    min_delta,
    checkpoint_path=None,
    history_csv_path=None,
):
    hidden_size = int(params["hidden_size"])
    num_layers = int(params["num_layers"])
    dropout = float(params["dropout"])
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])
    device = torch.device(device_str)

    model = MinimalLSTMNet(len(FEATURE_COLUMNS), hidden_size, num_layers, dropout, PRED_LEN).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x_train_t, y_train_t, x_val_t, y_val_t, pin_memory, preloaded_to_gpu = _make_lstm_tensors(
        x_train, y_train, x_val, y_val, device, preload_to_gpu=MINIMAL_LSTM_PRELOAD_TO_GPU
    )
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    best_val_loss, best_state, stale = np.inf, None, 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, count = 0.0, 0
        for x_b, y_b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if preloaded_to_gpu:
                x_device, y_device = x_b, y_b
            else:
                x_device = x_b.to(device, non_blocking=pin_memory)
                y_device = y_b.to(device, non_blocking=pin_memory)
            loss = criterion(model(x_device), y_device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            count += len(x_b)
        train_loss /= count
        val_loss = _lstm_eval_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if (best_val_loss - val_loss) > min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        history_rows.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss_so_far": float(best_val_loss),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
        })

        if stale >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience}, min_delta={min_delta})")
            break

    if history_csv_path is not None:
        from pathlib import Path as _Path

        history_path = _Path(history_csv_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(f"Saved history: {history_path}")

    if checkpoint_path is not None:
        from pathlib import Path as _Path
        _Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "pred_len": PRED_LEN,
            "context_len": CONTEXT_LEN,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "feature_columns": list(FEATURE_COLUMNS),
            "target_columns": list(TARGET_COLUMNS),
            "best_val_loss": float(best_val_loss),
        }, checkpoint_path)
        print(f"Saved: {checkpoint_path}  val_loss={best_val_loss:.6f}")

    return float(best_val_loss)


def make_hybrid_objective(cached_tracks, max_velocity_steps, rot_threshold_upper_bound=90.0):
    # creates an Optuna objective function for tuning the 4 hyperparameters of the HybridCVCTRVModel
    def objective(trial):
        cv_velocity_steps = trial.suggest_int("cv_velocity_steps", 1, max_velocity_steps)
        ctrv_velocity_steps = trial.suggest_int("ctrv_velocity_steps", 1, max_velocity_steps)
        rot_steps = trial.suggest_int("rot_steps", 1, max_velocity_steps)
        rot_threshold = trial.suggest_float("rot_threshold", 0.0, rot_threshold_upper_bound)
        model = HybridCVCTRVModel(
            cv_velocity_steps=cv_velocity_steps,
            ctrv_velocity_steps=ctrv_velocity_steps,
            rot_steps=rot_steps,
            rot_threshold=rot_threshold,
        )
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        return metrics["RMSE"]
    return objective


def make_kalman_objective(cached_tracks):
    def objective(trial):
        q_pos = trial.suggest_float("q_pos", 1e-3, 1e3, log=True)
        q_vel = trial.suggest_float("q_vel", 1e-5, 1e1, log=True)
        r_pos = trial.suggest_float("r_pos", 1e-2, 1e3, log=True)
        p0_pos = trial.suggest_float("p0_pos", 1e-2, 1e4, log=True)
        p0_vel = trial.suggest_float("p0_vel", 1e-4, 1e3, log=True)

        model = KalmanFilter(
            q_pos=q_pos,
            q_vel=q_vel,
            r_pos=r_pos,
            p0_pos=p0_pos,
            p0_vel=p0_vel,
        )
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        return metrics["RMSE"]

    return objective


def make_ctrv_ekf_objective(cached_tracks):
    max_ctx = max(len(track['x_ctx']) for track in cached_tracks)

    def objective(trial):
        q_pos = trial.suggest_float("q_pos", 1e-3, 1e3, log=True)
        q_vel = trial.suggest_float("q_vel", 1e-5, 1e1, log=True)
        q_rot = trial.suggest_float("q_rot", 1e-7, 1e1, log=True)
        r_pos = trial.suggest_float("r_pos", 1e-2, 1e3, log=True)
        p0_pos = trial.suggest_float("p0_pos", 1e-2, 1e4, log=True)
        p0_vel = trial.suggest_float("p0_vel", 1e-4, 1e3, log=True)
        p0_rot = trial.suggest_float("p0_rot", 1e-6, 1e2, log=True)

        init_velocity_steps = trial.suggest_int("init_velocity_steps", 1, max(1, max_ctx))
        model = CTRVExtendedKalmanFilter(
            q_pos=q_pos,
            q_vel=q_vel,
            q_rot=q_rot,
            r_pos=r_pos,
            p0_pos=p0_pos,
            p0_vel=p0_vel,
            p0_rot=p0_rot,
            init_velocity_steps=init_velocity_steps,
        )
        t0 = time.perf_counter()
        try:
            metrics = evaluate_model_cached_numpy(model, cached_tracks)
            rmse = float(metrics["RMSE"])
            elapsed = time.perf_counter() - t0
            if elapsed > 15:
                print(f"[CTRV EKF] trial {trial.number} slow: {elapsed:.1f}s  RMSE={rmse:.4f}")
            if not np.isfinite(rmse):
                return 1e12
            return rmse
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            # Keep optimization running for numerically unstable parameter sets.
            return 1e12

    return objective


class RMSEEarlyStoppingCallback:
    #early stopping 
    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = None
        self.stale_trials = 0

    def __call__(self, study, trial):
        if trial.value is None:
            return
        value = float(trial.value)
        if self.best_value is None or value < (self.best_value - self.min_delta):
            self.best_value = value
            self.stale_trials = 0
            return

        self.stale_trials += 1
        if self.stale_trials >= self.patience:
            study.stop()


class TrialProgressCallback:
    def __init__(self, model_label, total_trials):
        self.model_label = str(model_label)
        self.total_trials = int(total_trials)

    def __call__(self, study, trial):
        completed = len(study.trials)
        value = trial.value
        value_txt = f"{float(value):.6f}" if value is not None and np.isfinite(value) else "n/a"
        best_txt = "n/a"
        try:
            best_txt = f"{float(study.best_value):.6f}"
        except Exception:
            pass

        print(
            f"[Trial] {self.model_label}: {completed}/{self.total_trials} "
            f"(trial #{trial.number}) value={value_txt} best={best_txt}"
        )




def _load_branch_velocity_steps(diagnostics_dir):
    #reads the best velocity_steps found for CV and CTRV branches from the tuning summary CSV
    #to be used as seeds for the hybrid search
    df = pd.read_csv(diagnostics_dir / "tuning_best_params_val.csv")
    best_values = {
        str(row.get("model_key", "")).strip().lower(): int(row["best_velocity_steps"])
        for _, row in df.iterrows()
        if str(row.get("model_key", "")).strip() and pd.notna(row.get("best_velocity_steps"))
    }
    cv_velocity_steps = best_values.get("cv", best_values.get("constant_velocity"))
    ctrv_velocity_steps = best_values.get("ctrv")
    return cv_velocity_steps, ctrv_velocity_steps


def _load_existing_best_rows(diagnostics_dir):
    #loads existing best rows , to be used for seeding the hybrid search 
    # if CV/CTRV tuning is not run in the current execution
    return pd.read_csv(diagnostics_dir / "tuning_best_params_val.csv").to_dict(orient="records")


def run_grid_search_for_velocity_model(model_key, model_label, model_cls, data_dir, diagnostics_dir, model_kwargs=None):
    # Full grid search over velocity_steps from 1..max_velocity_steps.
    cached_tracks = load_tracks_cached_numpy(
        data_dir,
        split='val',
        context_filter=CONTEXT_FILTER_TUNING,
        sample_pct=TUNING_VALIDATION_PCT,
        seed=SEED,
    )

    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))
    model_kwargs = dict(model_kwargs or {})

    rows = []
    best_steps = 1
    best_rmse = float("inf")

    for i, velocity_steps in enumerate(range(1, max_velocity_steps + 1), start=1):
        model = model_cls(velocity_steps=velocity_steps, **model_kwargs)
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        rmse = float(metrics["RMSE"])

        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_steps = int(velocity_steps)

        value_txt = f"{rmse:.6f}" if np.isfinite(rmse) else "n/a"
        best_txt = f"{best_rmse:.6f}" if np.isfinite(best_rmse) else "n/a"
        print(
            f"[Grid] {model_label}: {i}/{max_velocity_steps} "
            f"(velocity_steps={velocity_steps}) value={value_txt} best={best_txt}"
        )

        rows.append({
            "number": i - 1,
            "value": rmse,
            "params_velocity_steps": int(velocity_steps),
            "state": "COMPLETE",
        })

    trials_df = pd.DataFrame(rows)
    trials_df["model_key"] = model_key
    trials_df["model_label"] = model_label
    trials_df["sample_pct"] = TUNING_VALIDATION_PCT

    csv_path = diagnostics_dir / f"tuning_{model_key}_val.csv"
    trials_df.to_csv(csv_path, index=False)

    return {
        "model_key": model_key,
        "model_label": model_label,
        "best_velocity_steps": int(best_steps),
        "best_val_rmse": float(best_rmse),
        "trials_csv": str(csv_path),
        "sample_pct": TUNING_VALIDATION_PCT,
    }, trials_df


def run_hybrid_optimization(data_dir, diagnostics_dir):

    #1. Load validation tracks into memory
    #2. Create Optuna study and optimize the objective function with TPE, early stopping
    #   and multivariate sampling
    #3. Seeding the hybrid search with the best velocity_steps found for CV and CTRV branches, if available
    #4. Save all trials and best parameters to CSV
    #5. Return best parameters for summary table


    cached_tracks = load_tracks_cached_numpy(
        data_dir,
        split='val',
        context_filter=CONTEXT_FILTER_TUNING,
        sample_pct=TUNING_VALIDATION_PCT,
        seed=SEED,
    )

    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))

    seed_cv_steps, seed_ctrv_steps = _load_branch_velocity_steps(diagnostics_dir)

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20, n_ei_candidates=100, multivariate=False)
    study = optuna.create_study(
        study_name="hybrid_cv_ctrv_val_tpe",
        direction="minimize",
        sampler=sampler,
    )
    study.enqueue_trial({ #seed the search with the best velocity_steps found for CV and CTRV branches
        "cv_velocity_steps": seed_cv_steps,
        "ctrv_velocity_steps": seed_ctrv_steps,
        "rot_steps": seed_ctrv_steps,
        "rot_threshold": 1.0,
    })

    study.optimize(
        make_hybrid_objective(cached_tracks, max_velocity_steps=max_velocity_steps),
        n_trials=HYBRID_N_TRIALS,
        n_jobs=1,
        callbacks=[
            RMSEEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA),
            TrialProgressCallback(model_label="Hybrid CV/CTRV", total_trials=HYBRID_N_TRIALS),
        ],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "hybrid_cv_ctrv"
    trials_df["model_label"] = "Hybrid CV/CTRV"
    trials_df["sample_pct"] = TUNING_VALIDATION_PCT

    csv_path = diagnostics_dir / "tuning_hybrid_cv_ctrv_val.csv"
    trials_df.to_csv(csv_path, index=False)

    best = study.best_trial
    best_cv_steps = int(best.params["cv_velocity_steps"])
    best_ctrv_steps = int(best.params["ctrv_velocity_steps"])
    best_rot_steps = int(best.params["rot_steps"])
    best_threshold = float(best.params["rot_threshold"])
    best_rmse = float(best.value)

    return {
        "model_key": "hybrid_cv_ctrv",
        "model_label": "Hybrid CV/CTRV",
        "cv_velocity_steps": best_cv_steps,
        "ctrv_velocity_steps": best_ctrv_steps,
        "best_rot_steps": best_rot_steps,
        "best_rot_threshold": best_threshold,
        "best_val_rmse": best_rmse,
        "trials_csv": str(csv_path),
        "sample_pct": TUNING_VALIDATION_PCT,
    }, trials_df


def run_kalman_optimization(data_dir, diagnostics_dir):
    cached_tracks = load_tracks_cached_numpy(
        data_dir,
        split='val',
        context_filter=CONTEXT_FILTER_TUNING,
        sample_pct=TUNING_VALIDATION_PCT,
        seed=SEED,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20, n_ei_candidates=100, multivariate=False)
    study = optuna.create_study(
        study_name="kalman_val_tpe",
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        make_kalman_objective(cached_tracks),
        n_trials=KALMAN_N_TRIALS,
        n_jobs=1,
        callbacks=[
            RMSEEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA),
            TrialProgressCallback(model_label="Kalman", total_trials=KALMAN_N_TRIALS),
        ],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "kalman"
    trials_df["model_label"] = "Kalman"
    trials_df["sample_pct"] = TUNING_VALIDATION_PCT

    csv_path = diagnostics_dir / "tuning_kalman_val.csv"
    trials_df.to_csv(csv_path, index=False)

    best = study.best_trial
    return {
        "model_key": "kalman",
        "model_label": "Kalman",
        "best_q_pos": float(best.params["q_pos"]),
        "best_q_vel": float(best.params["q_vel"]),
        "best_r_pos": float(best.params["r_pos"]),
        "best_p0_pos": float(best.params["p0_pos"]),
        "best_p0_vel": float(best.params["p0_vel"]),
        "best_val_rmse": float(best.value),
        "trials_csv": str(csv_path),
        "sample_pct": TUNING_VALIDATION_PCT,
    }, trials_df


def run_ctrv_ekf_optimization(data_dir, diagnostics_dir):
    cached_tracks = load_tracks_cached_numpy(
        data_dir,
        split='val',
        context_filter=CONTEXT_FILTER_TUNING,
        sample_pct=TUNING_VALIDATION_PCT,
        seed=SEED,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20, n_ei_candidates=100, multivariate=False)
    study = optuna.create_study(
        study_name="ctrv_ekf_val_tpe",
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        make_ctrv_ekf_objective(cached_tracks),
        n_trials=CTRV_EKF_N_TRIALS,
        n_jobs=1,
        callbacks=[
            RMSEEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA),
            TrialProgressCallback(model_label="CTRV EKF", total_trials=CTRV_EKF_N_TRIALS),
        ],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "ctrv_ekf"
    trials_df["model_label"] = "CTRV EKF"
    trials_df["sample_pct"] = TUNING_VALIDATION_PCT

    csv_path = diagnostics_dir / "tuning_ctrv_ekf_val.csv"
    trials_df.to_csv(csv_path, index=False)

    best = study.best_trial
    return {
        "model_key": "ctrv_ekf",
        "model_label": "CTRV EKF",
        "best_q_pos": float(best.params["q_pos"]),
        "best_q_vel": float(best.params["q_vel"]),
        "best_q_rot": float(best.params["q_rot"]),
        "best_r_pos": float(best.params["r_pos"]),
        "best_p0_pos": float(best.params["p0_pos"]),
        "best_p0_vel": float(best.params["p0_vel"]),
        "best_p0_rot": float(best.params["p0_rot"]),
        "best_val_rmse": float(best.value),
        "trials_csv": str(csv_path),
        "best_init_velocity_steps": int(best.params["init_velocity_steps"]),
        "sample_pct": TUNING_VALIDATION_PCT,
    }, trials_df


def run_minimal_lstm_optimization(diagnostics_dir):
    tuning_cfg = MINIMAL_LSTM_TUNING_CFG

    x_train, y_train = _build_lstm_samples("train", sample_pct=TUNING_VALIDATION_PCT, seed=SEED)
    x_val, y_val = _build_lstm_samples("val", sample_pct=TUNING_VALIDATION_PCT, seed=SEED)
    print(f"Minimal LSTM tuning: {x_train.shape[0]} train samples, {x_val.shape[0]} val samples")

    def objective(trial):
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", tuning_cfg["hidden_size"]),
            "num_layers": trial.suggest_categorical("num_layers", tuning_cfg["num_layers"]),
            "dropout": trial.suggest_float("dropout", tuning_cfg["dropout_min"], tuning_cfg["dropout_max"]),
            "learning_rate": trial.suggest_float("learning_rate", tuning_cfg["learning_rate_min"], tuning_cfg["learning_rate_max"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", tuning_cfg["batch_size"]),
        }
        history_path = diagnostics_dir / "minimal_lstm_history" / f"minimal_lstm_trial_{trial.number:04d}.csv"
        return _run_lstm_training(
            params, x_train, y_train, x_val, y_val,
            device_str=MINIMAL_LSTM_DEVICE,
            max_epochs=MINIMAL_LSTM_MAX_EPOCHS,
            patience=MINIMAL_LSTM_PATIENCE,
            min_delta=MINIMAL_LSTM_MIN_DELTA,
            history_csv_path=history_path,
        )

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10, multivariate=False)
    study = optuna.create_study(study_name="minimal_lstm_val_tpe", direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=MINIMAL_LSTM_N_TRIALS,
        n_jobs=1,
        callbacks=[
            RMSEEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA),
            TrialProgressCallback(model_label="Minimal LSTM", total_trials=MINIMAL_LSTM_N_TRIALS),
        ],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "minimal_lstm"
    trials_df["model_label"] = "Minimal LSTM"
    trials_df["sample_pct"] = TUNING_VALIDATION_PCT
    csv_path = diagnostics_dir / "tuning_minimal_lstm_val.csv"
    trials_df.to_csv(csv_path, index=False)

    best = study.best_trial
    _run_lstm_training(
        best.params, x_train, y_train, x_val, y_val,
        device_str=MINIMAL_LSTM_DEVICE,
        max_epochs=MINIMAL_LSTM_MAX_EPOCHS,
        patience=MINIMAL_LSTM_PATIENCE,
        min_delta=MINIMAL_LSTM_MIN_DELTA,
        checkpoint_path=PROJECT_ROOT / MINIMAL_LSTM_CHECKPOINT,
        history_csv_path=diagnostics_dir / "minimal_lstm_history" / "minimal_lstm_best_retrain.csv",
    )

    return {
        "model_key": "minimal_lstm",
        "model_label": "Minimal LSTM",
        "best_hidden_size": int(best.params["hidden_size"]),
        "best_num_layers": int(best.params["num_layers"]),
        "best_dropout": float(best.params["dropout"]),
        "best_learning_rate": float(best.params["learning_rate"]),
        "best_batch_size": int(best.params["batch_size"]),
        "best_val_loss": float(best.value),
        "trials_csv": str(csv_path),
        "sample_pct": TUNING_VALIDATION_PCT,
    }, trials_df


def _run_model_job(result_queue, model_key, model_label, model_cls, data_dir, diagnostics_dir, model_kwargs=None):
    # mp.Process target: runs 1D velocity model tuning in a separate process
    # and puts (best_row, trials_df) into the shared queue for the main process to collect
    # necessary for multiprocessing
    best_row, trials_df = run_grid_search_for_velocity_model(
        model_key=model_key,
        model_label=model_label,
        model_cls=model_cls,
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
        model_kwargs=model_kwargs,
    )
    result_queue.put((best_row, trials_df))


def _run_hybrid_job(result_queue, data_dir, diagnostics_dir):
    # mp.Process target: runs hybrid tuning in a separate process
    # and puts (best_row, trials_df) into the shared queue for the main process to collect
    best_row, trials_df = run_hybrid_optimization(
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
    )
    result_queue.put((best_row, trials_df))


def _run_kalman_job(result_queue, data_dir, diagnostics_dir):
    best_row, trials_df = run_kalman_optimization(
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
    )
    result_queue.put((best_row, trials_df))


def _run_ctrv_ekf_job(result_queue, data_dir, diagnostics_dir):
    best_row, trials_df = run_ctrv_ekf_optimization(
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
    )
    result_queue.put((best_row, trials_df))


if __name__ == "__main__":
    # 1. Set up paths and output directories
    # 2. Launch CV and CTRV tuning as parallel processes
    # 3. Wait for both, collect results, save intermediate best-params CSV
    # 4. Run hybrid tuning sequentially (needs CV/CTRV seeds from step 3)
    # 5. Write final best-params and all-trials CSVs
    import multiprocessing as mp

    run_paths = resolve_run_paths(PROJECT_ROOT, CONFIG, create=True)
    data_dir = PROJECT_ROOT / CONFIG["data"]["parquet_dir"]
    diagnostics_dir = run_paths["tuning_dir"]

    existing_tuning_csvs = list(diagnostics_dir.glob("tuning_*_val.csv")) + [diagnostics_dir / "tuning_best_params_val.csv", diagnostics_dir / "tuning_all_trials_val.csv"]
    if any(p.exists() for p in existing_tuning_csvs):
        print(f"[Warning] Existing tuning outputs found in {diagnostics_dir}. Files will be overwritten for run '{CONFIG['run']['name']}'.")

    print(f"Hybrid search trials: {HYBRID_N_TRIALS} | Kalman: {KALMAN_N_TRIALS} | CTRV EKF: {CTRV_EKF_N_TRIALS}")
    print(
        f"Run: {CONFIG['run']['name']} | seed={SEED} | sample_pct={SAMPLE_PCT}% "
        f"| tuning_validation_pct={TUNING_VALIDATION_PCT}%"
    )
    print("Validation tracks are cached once in RAM per model process\n")

    standard_jobs = []
    if RUN_CV:
        standard_jobs.append(("cv", "Constant Velocity", ConstantVelocityModel, {}))
    if RUN_CTRV:
        standard_jobs.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel, {}))
    if RUN_CTRV_ARC:
        standard_jobs.append(("ctrv_arc", "CTRV Arc", ConstantTurnRateVelocityArcModel, {}))
    if RUN_TIREX_LSTM:
        standard_jobs.append((
            "tirex_lstm",
            "TiRex LSTM",
            TirexLSTMModel,
            {
                "model_name": TIREX_MODEL_NAME,
                "device": TIREX_DEVICE,
                "backend": TIREX_BACKEND,
                "compile_model": TIREX_COMPILE_MODEL,
                "batch_size": TIREX_BATCH_SIZE,
                "scaler_path": str(PROJECT_ROOT / TIREX_SCALER_PATH),
            },
        ))

    if not standard_jobs and not RUN_HYBRID and not RUN_KALMAN and not RUN_CTRV_EKF and not RUN_MINIMAL_LSTM:
        print("No models selected. Set RUN_CV and/or RUN_CTRV and/or RUN_CTRV_ARC and/or RUN_TIREX_LSTM and/or RUN_HYBRID and/or RUN_KALMAN and/or RUN_CTRV_EKF and/or RUN_MINIMAL_LSTM to True.")
        raise SystemExit(0)

    best_rows = []
    all_trials = []

    total_steps = (
        len(standard_jobs)
        + (1 if RUN_KALMAN else 0)
        + (1 if RUN_CTRV_EKF else 0)
        + (1 if RUN_HYBRID else 0)
        + (1 if RUN_MINIMAL_LSTM else 0)
    )
    completed_steps = 0

    result_queue = mp.Queue()
    processes = []

    if RUN_HYBRID and not standard_jobs:
        best_rows = _load_existing_best_rows(diagnostics_dir)
        print("Using existing tuning_best_params_val.csv for CV/CTRV hybrid seeding.")

    for model_key, model_label, model_cls, model_kwargs in standard_jobs:
        p = mp.Process(
            target=_run_model_job,
            args=(
                result_queue,
                model_key,
                model_label,
                model_cls,
                data_dir,
                diagnostics_dir,
                model_kwargs,
            ),
        )
        p.start()
        processes.append(p)

    if RUN_KALMAN:
        p = mp.Process(
            target=_run_kalman_job,
            args=(
                result_queue,
                data_dir,
                diagnostics_dir,
            ),
        )
        p.start()
        processes.append(p)

    if RUN_CTRV_EKF:
        p = mp.Process(
            target=_run_ctrv_ekf_job,
            args=(
                result_queue,
                data_dir,
                diagnostics_dir,
            ),
        )
        p.start()
        processes.append(p)

    expected_results = len(standard_jobs) + (1 if RUN_KALMAN else 0) + (1 if RUN_CTRV_EKF else 0)
    for _ in range(expected_results):
        best_row, trials_df = result_queue.get()
        best_rows.append(best_row)
        all_trials.append(trials_df)
        completed_steps += 1
        print(f"[Progress] {completed_steps}/{total_steps} steps completed ({best_row.get('model_label', best_row.get('model_key', 'unknown'))})")

    for p in processes:
        p.join()

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_path = diagnostics_dir / "tuning_best_params_val.csv"
        best_df.to_csv(best_path, index=False)
        print(f"\nBest-parameter summary saved to {best_path}")

    if RUN_HYBRID:
        hybrid_process = mp.Process(
            target=_run_hybrid_job,
            args=(
                result_queue,
                data_dir,
                diagnostics_dir,
            ),
        )
        hybrid_process.start()
        best_row, trials_df = result_queue.get()
        hybrid_process.join()
        best_rows = [row for row in best_rows if row.get("model_key") != best_row["model_key"]]
        best_rows.append(best_row)
        all_trials.append(trials_df)
        completed_steps += 1
        print(f"[Progress] {completed_steps}/{total_steps} steps completed ({best_row.get('model_label', best_row.get('model_key', 'unknown'))})")

    if RUN_MINIMAL_LSTM:
        best_row, trials_df = run_minimal_lstm_optimization(diagnostics_dir)
        best_rows = [row for row in best_rows if row.get("model_key") != "minimal_lstm"]
        best_rows.append(best_row)
        all_trials.append(trials_df)
        completed_steps += 1
        print(f"[Progress] {completed_steps}/{total_steps} steps completed (Minimal LSTM)")

    best_df = pd.DataFrame(best_rows)
    best_path = diagnostics_dir / "tuning_best_params_val.csv"
    best_df.to_csv(best_path, index=False)
    print(f"\nBest-parameter summary saved to {best_path}")

    all_trials_df = pd.concat(all_trials, ignore_index=True)
    all_trials_path = diagnostics_dir / "tuning_all_trials_val.csv"
    all_trials_df.to_csv(all_trials_path, index=False)
    print(f"All trials table saved to {all_trials_path}")

    write_run_metadata(run_paths, CONFIG, stage="tuning")
    update_latest_run_pointer(run_paths)

    print("\nNext step: use the saved best parameters for the final test evaluation.")
