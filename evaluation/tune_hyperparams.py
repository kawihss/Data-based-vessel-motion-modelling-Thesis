import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, HybridCVCTRVModel
from models.filters import KalmanFilter
from evaluation.evaluator import load_tracks_cached_numpy, evaluate_model_cached_numpy

CONTEXT_FILTER = None   # None = all contexts; or e.g. 'lock', or ['harbour', 'lock'] 
RUN_CV = False
RUN_CTRV = False
RUN_HYBRID = False
RUN_KALMAN = True
N_TRIALS = 15
HYBRID_N_TRIALS = 200
KALMAN_N_TRIALS = 80


def make_objective(model_cls, cached_tracks, max_velocity_steps):
    # creates an Optuna objective function for tuning velocity_steps of a given model 

    # use closures to pass the model class and cached tracks to the objective function 
    # as only 1 argument (trial) is allowed by Optuna
    def objective(trial):
        velocity_steps = trial.suggest_int("velocity_steps", 1, max_velocity_steps)
        model = model_cls(velocity_steps=velocity_steps)
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        return metrics["RMSE"]
    return objective


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


class RMSEEarlyStoppingCallback:
    #early stopping 
    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = None
        self.stale_trials = 0

    def __call__(self, study, trial):
        value = float(trial.value)
        if self.best_value is None or value < (self.best_value - self.min_delta):
            self.best_value = value
            self.stale_trials = 0
            return

        self.stale_trials += 1
        if self.stale_trials >= self.patience:
            study.stop()


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


def run_optimization_for_model(model_key, model_label, model_cls, data_dir, diagnostics_dir):

    #1. Load validation tracks into memory
    #2. Create Optuna study and optimize the objective function with TPE and early stopping
    # (TPE not the most efficient, but used here to demonstrate how to use Optuna
    #3. Save all trials and best parameters to CSV
    #4. Return best parameters for summary table

    cached_tracks = load_tracks_cached_numpy(data_dir, split='val', context_filter=CONTEXT_FILTER)
    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5, n_ei_candidates=100)
    study = optuna.create_study(
        study_name=f"{model_key}_val_gridsearch",
        direction="minimize",
        sampler=sampler,
    )

    total = N_TRIALS
    study.optimize(
        make_objective(model_cls, cached_tracks, max_velocity_steps),
        n_trials=total,
        n_jobs=1,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = model_key
    trials_df["model_label"] = model_label

    csv_path = diagnostics_dir / f"tuning_{model_key}_val.csv"
    trials_df.to_csv(csv_path, index=False)

    best = study.best_trial
    best_steps = int(best.params["velocity_steps"])
    best_rmse = float(best.value)

    return {
        "model_key": model_key,
        "model_label": model_label,
        "best_velocity_steps": best_steps,
        "best_val_rmse": best_rmse,
        "trials_csv": str(csv_path),
    }, trials_df


def run_hybrid_optimization(data_dir, diagnostics_dir):

    #1. Load validation tracks into memory
    #2. Create Optuna study and optimize the objective function with TPE, early stopping
    #   and multivariate sampling
    #3. Seeding the hybrid search with the best velocity_steps found for CV and CTRV branches, if available
    #4. Save all trials and best parameters to CSV
    #5. Return best parameters for summary table


    cached_tracks = load_tracks_cached_numpy(data_dir, split='val', context_filter=CONTEXT_FILTER)

    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))

    seed_cv_steps, seed_ctrv_steps = _load_branch_velocity_steps(diagnostics_dir)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20, n_ei_candidates=100, multivariate=True)
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
        callbacks=[RMSEEarlyStoppingCallback(patience=20, min_delta=1e-4)],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "hybrid_cv_ctrv"
    trials_df["model_label"] = "Hybrid CV/CTRV"

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
    }, trials_df


def run_kalman_optimization(data_dir, diagnostics_dir):
    cached_tracks = load_tracks_cached_numpy(data_dir, split='val', context_filter=CONTEXT_FILTER)

    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20, n_ei_candidates=100, multivariate=True)
    study = optuna.create_study(
        study_name="kalman_val_tpe",
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        make_kalman_objective(cached_tracks),
        n_trials=KALMAN_N_TRIALS,
        n_jobs=1,
        callbacks=[RMSEEarlyStoppingCallback(patience=20, min_delta=1e-4)],
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "kalman"
    trials_df["model_label"] = "Kalman"

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
    }, trials_df

def _run_model_job(result_queue, model_key, model_label, model_cls, data_dir, diagnostics_dir):
    # mp.Process target: runs CV/CTRV tuning in a separate process
    # and puts (best_row, trials_df) into the shared queue for the main process to collect
    # necessary for multiprocessing
    best_row, trials_df = run_optimization_for_model(
        model_key=model_key,
        model_label=model_label,
        model_cls=model_cls,
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
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


if __name__ == "__main__":
    # 1. Set up paths and output directories
    # 2. Launch CV and CTRV tuning as parallel processes
    # 3. Wait for both, collect results, save intermediate best-params CSV
    # 4. Run hybrid tuning sequentially (needs CV/CTRV seeds from step 3)
    # 5. Write final best-params and all-trials CSVs
    import multiprocessing as mp

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "output" / "07_parquet"
    diagnostics_dir = project_root / "evaluation" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tuning velocity_steps with {N_TRIALS} trial(s) for CV/CTRV and 3D hybrid search with {HYBRID_N_TRIALS} trial(s)")
    print("Validation tracks are cached once in RAM per model process\n")

    standard_jobs = []
    if RUN_CV:
        standard_jobs.append(("cv", "Constant Velocity", ConstantVelocityModel))
    if RUN_CTRV:
        standard_jobs.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel))

    if not standard_jobs and not RUN_HYBRID and not RUN_KALMAN:
        print("No models selected. Set RUN_CV and/or RUN_CTRV and/or RUN_HYBRID and/or RUN_KALMAN to True.")
        raise SystemExit(0)

    best_rows = []
    all_trials = []

    result_queue = mp.Queue()
    processes = []

    if RUN_HYBRID and not standard_jobs:
        best_rows = _load_existing_best_rows(diagnostics_dir)
        print("Using existing tuning_best_params_val.csv for CV/CTRV hybrid seeding.")

    for model_key, model_label, model_cls in standard_jobs:
        p = mp.Process(
            target=_run_model_job,
            args=(
                result_queue,
                model_key,
                model_label,
                model_cls,
                data_dir,
                diagnostics_dir,
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

    for p in processes:
        p.join()

    expected_results = len(standard_jobs) + (1 if RUN_KALMAN else 0)
    for _ in range(expected_results):
        best_row, trials_df = result_queue.get()
        best_rows.append(best_row)
        all_trials.append(trials_df)

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
        hybrid_process.join()

        best_row, trials_df = result_queue.get()
        best_rows = [row for row in best_rows if row.get("model_key") != best_row["model_key"]]
        best_rows.append(best_row)
        all_trials.append(trials_df)

    best_df = pd.DataFrame(best_rows)
    best_path = diagnostics_dir / "tuning_best_params_val.csv"
    best_df.to_csv(best_path, index=False)
    print(f"\nBest-parameter summary saved to {best_path}")

    all_trials_df = pd.concat(all_trials, ignore_index=True)
    all_trials_path = diagnostics_dir / "tuning_all_trials_val.csv"
    all_trials_df.to_csv(all_trials_path, index=False)
    print(f"All trials table saved to {all_trials_path}")

    print("\nNext step: use the saved best parameters for the final test evaluation.")
