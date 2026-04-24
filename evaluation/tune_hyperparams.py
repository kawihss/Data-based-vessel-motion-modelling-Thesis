import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel, HybridCVCTRVModel
from evaluation.evaluator import load_tracks_cached_numpy, evaluate_model_cached_numpy

CONTEXT_FILTER = None   # None = all contexts; or e.g. 'lock'
RUN_CV = False
RUN_CTRV = False
RUN_HYBRID = True
N_TRIALS = 15
HYBRID_N_TRIALS = 200
ENABLE_TIMING_DIAGNOSTICS = True


def _safe_print(print_lock, *args, **kwargs):
    if print_lock is None:
        print(*args, **kwargs)
        return
    with print_lock:
        print(*args, **kwargs)


def make_objective(model_cls, cached_tracks, max_velocity_steps, timing_stats=None):
    def objective(trial):
        velocity_steps = trial.suggest_int("velocity_steps", 1, max_velocity_steps)
        model = model_cls(velocity_steps=velocity_steps)
        t0 = time.perf_counter()
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        if timing_stats is not None:
            timing_stats['eval_calls'] += 1
            timing_stats['eval_s'] += time.perf_counter() - t0
        return metrics["RMSE"]
    return objective


def make_hybrid_objective(cached_tracks, max_velocity_steps, timing_stats=None, rot_threshold_upper_bound=90.0):
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
        t0 = time.perf_counter()
        metrics = evaluate_model_cached_numpy(model, cached_tracks)
        if timing_stats is not None:
            timing_stats['eval_calls'] += 1
            timing_stats['eval_s'] += time.perf_counter() - t0
        return metrics["RMSE"]
    return objective


class RMSEEarlyStoppingCallback:
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
    best_path = diagnostics_dir / "tuning_best_params_val.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Hybrid tuning seed file not found: {best_path}")

    df = pd.read_csv(best_path)
    required_cols = {"model_key", "best_velocity_steps", "best_val_rmse"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Hybrid tuning seed file missing columns: {missing_cols}")

    best_values = {}
    for _, row in df.iterrows():
        key = str(row.get("model_key", "")).strip().lower()
        value = row.get("best_velocity_steps")
        if key and pd.notna(value):
            best_values[key] = int(value)

    cv_velocity_steps = best_values.get("cv", best_values.get("constant_velocity"))
    ctrv_velocity_steps = best_values.get("ctrv")
    if cv_velocity_steps is None or ctrv_velocity_steps is None:
        raise ValueError("Hybrid tuning requires best_velocity_steps for both CV and CTRV.")

    return cv_velocity_steps, ctrv_velocity_steps


def _load_existing_best_rows(diagnostics_dir):
    best_path = diagnostics_dir / "tuning_best_params_val.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Existing tuning summary not found: {best_path}")

    df = pd.read_csv(best_path)
    if df.empty:
        raise ValueError(f"Existing tuning summary is empty: {best_path}")
    if "model_key" not in df.columns:
        raise ValueError("Existing tuning summary missing 'model_key' column.")

    return df.to_dict(orient="records")


def plot_results(df, best_steps, best_rmse, model_label, output_path, print_fn=print):
    df_sorted = df.sort_values("params_velocity_steps")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        df_sorted["params_velocity_steps"],
        df_sorted["value"],
        marker="o",
        linewidth=1.8,
        markersize=5,
        color="steelblue",
        label="Val RMSE",
    )
    ax.axvline(best_steps, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Best velocity_steps={best_steps}  (RMSE={best_rmse:.4f} m)")
    ax.annotate(
        f"velocity_steps={best_steps}\nRMSE={best_rmse:.4f} m",
        xy=(best_steps, best_rmse),
        xytext=(best_steps + 0.2, best_rmse + (df_sorted["value"].max() - df_sorted["value"].min()) * 0.08),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        fontsize=9,
        color="crimson",
    )
    ax.set_xlabel("velocity_steps  (number of recent context steps averaged)", fontsize=11)
    ax.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax.set_title(f"{model_label} - Hyperparameter Optimization\nVal-split RMSE vs velocity_steps", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print_fn(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_hybrid_results(
    df,
    best_cv_steps,
    best_ctrv_steps,
    best_rot_steps,
    best_threshold,
    best_rmse,
    model_label,
    output_path,
    print_fn=print,
):
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: best RMSE observed for each (cv_steps, ctrv_steps) pair.
    pair_best = (
        df.groupby(["params_cv_velocity_steps", "params_ctrv_velocity_steps"], as_index=False)["value"]
        .min()
        .rename(columns={"value": "min_rmse"})
    )
    heat = pair_best.pivot(
        index="params_ctrv_velocity_steps",
        columns="params_cv_velocity_steps",
        values="min_rmse",
    ).sort_index().sort_index(axis=1)

    im = ax_left.imshow(heat.values, origin="lower", aspect="auto", cmap="viridis_r")
    ax_left.set_xticks(range(len(heat.columns)))
    ax_left.set_xticklabels([int(v) for v in heat.columns])
    ax_left.set_yticks(range(len(heat.index)))
    ax_left.set_yticklabels([int(v) for v in heat.index])
    ax_left.set_xlabel("cv_velocity_steps", fontsize=11)
    ax_left.set_ylabel("ctrv_velocity_steps", fontsize=11)
    ax_left.set_title("Best RMSE per step-pair", fontsize=12)

    if best_cv_steps in heat.columns and best_ctrv_steps in heat.index:
        x_idx = list(heat.columns).index(best_cv_steps)
        y_idx = list(heat.index).index(best_ctrv_steps)
        ax_left.scatter([x_idx], [y_idx], marker="*", s=220, color="crimson", edgecolors="white", linewidths=1.0)

    cbar = fig.colorbar(im, ax=ax_left)
    cbar.set_label("Validation RMSE  [m]")

    # Panel 2: RMSE vs rot_steps for all trials (scatter), best point starred.
    ax_mid.scatter(
        df["params_rot_steps"],
        df["value"],
        s=25,
        alpha=0.5,
        color="steelblue",
        label="All trials",
    )
    ax_mid.scatter(
        [best_rot_steps],
        [best_rmse],
        color="crimson",
        marker="*",
        s=220,
        zorder=5,
        label=f"Best rot_steps={best_rot_steps}\nRMSE={best_rmse:.4f} m",
    )
    ax_mid.set_xlabel("rot_steps", fontsize=11)
    ax_mid.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax_mid.set_title("RMSE vs rot_steps (all trials)", fontsize=12)
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(fontsize=9)

    # Panel 3: RMSE vs rot_threshold for all trials (scatter), best point starred.
    ax_right.scatter(
        df["params_rot_threshold"],
        df["value"],
        s=25,
        alpha=0.5,
        color="steelblue",
        label="All trials",
    )
    ax_right.scatter(
        [best_threshold],
        [best_rmse],
        color="crimson",
        marker="*",
        s=220,
        zorder=5,
        label=f"Best rot_threshold={best_threshold:.4f}\nRMSE={best_rmse:.4f} m",
    )
    ax_right.set_xlabel("rot_threshold  [deg/min]", fontsize=11)
    ax_right.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax_right.set_title("RMSE vs rot_threshold (all trials)", fontsize=12)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=9)

    fig.suptitle(f"{model_label} - Hyperparameter Optimization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print_fn(f"Plot saved to {output_path}")
    plt.close(fig)


def run_optimization_for_model(model_key, model_label, model_cls, data_dir, diagnostics_dir, print_lock=None):
    print_fn = lambda *args, **kwargs: _safe_print(print_lock, *args, **kwargs)

    print_fn(f"\n=== {model_label} ===")
    t_load_start = time.perf_counter()
    cached_tracks = load_tracks_cached_numpy(data_dir, split='val', context_filter=CONTEXT_FILTER)
    load_s = time.perf_counter() - t_load_start
    if not cached_tracks:
        raise ValueError("No tracks available in cache for the selected split/context.")
    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))
    print_fn(f"Cached objective with velocity_steps in [1, {max_velocity_steps}]")

    timing_stats = {'eval_calls': 0, 'eval_s': 0.0}

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5, n_ei_candidates=100)
    study = optuna.create_study(
        study_name=f"{model_key}_val_gridsearch",
        direction="minimize",
        sampler=sampler,
    )

    total = N_TRIALS
    t_trials_start = time.perf_counter()
    for i in range(1, total + 1):
        study.optimize(
            make_objective(model_cls, cached_tracks, max_velocity_steps, timing_stats=timing_stats),
            n_trials=1,
            n_jobs=1,
        )
        last = study.trials[-1]
        trial_steps = int(last.params["velocity_steps"])
        print_fn(
            f"[{model_key.upper()}] Trial {i:>2}/{total}  "
            f"velocity_steps={trial_steps}  RMSE={last.value:.4f} m"
        )
    trials_s = time.perf_counter() - t_trials_start

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = model_key
    trials_df["model_label"] = model_label

    csv_path = diagnostics_dir / f"tuning_{model_key}_val.csv"
    trials_df.to_csv(csv_path, index=False)
    print_fn(f"\nTrial table saved to {csv_path}")

    rmse_counts = trials_df["value"].round(6).value_counts()
    repeated = rmse_counts[rmse_counts > 1]
    if not repeated.empty:
        print_fn("\nRepeated RMSE values (rounded to 6 dp):")
        for rmse_value, count in repeated.items():
            print_fn(f"  RMSE={rmse_value:.6f} appears {int(count)}x")

    best = study.best_trial
    best_steps = int(best.params["velocity_steps"])
    best_rmse = float(best.value)

    print_fn(f"\n{'='*50}")
    print_fn(f"  Model                 : {model_label}")
    print_fn(f"  Best velocity_steps   : {best_steps}")
    print_fn(f"  Best val RMSE         : {best_rmse:.4f} m")
    print_fn(f"{'='*50}")

    if ENABLE_TIMING_DIAGNOSTICS:
        eval_calls = max(1, int(timing_stats['eval_calls']))
        avg_eval_ms = (timing_stats['eval_s'] / eval_calls) * 1000.0
        overhead_s = max(0.0, trials_s - timing_stats['eval_s'])
        print_fn("\nTiming diagnostics:")
        print_fn(f"  Cache load time        : {load_s:.3f} s")
        print_fn(f"  Trial loop total       : {trials_s:.3f} s")
        print_fn(f"  Pure eval time         : {timing_stats['eval_s']:.3f} s")
        print_fn(f"  Avg eval per trial     : {avg_eval_ms:.1f} ms")
        print_fn(f"  Sampler/loop overhead  : {overhead_s:.3f} s")

    plot_path = diagnostics_dir / f"tuning_{model_key}_val_plot.png"
    plot_results(
        df=trials_df,
        best_steps=best_steps,
        best_rmse=best_rmse,
        model_label=model_label,
        output_path=plot_path,
        print_fn=print_fn,
    )

    return {
        "model_key": model_key,
        "model_label": model_label,
        "best_velocity_steps": best_steps,
        "best_val_rmse": best_rmse,
        "trials_csv": str(csv_path),
        "plot_png": str(plot_path),
    }, trials_df


def run_hybrid_optimization(data_dir, diagnostics_dir, print_lock=None):
    print_fn = lambda *args, **kwargs: _safe_print(print_lock, *args, **kwargs)

    print_fn("\n=== Hybrid CV/CTRV ===")
    t_load_start = time.perf_counter()
    cached_tracks = load_tracks_cached_numpy(data_dir, split='val', context_filter=CONTEXT_FILTER)
    load_s = time.perf_counter() - t_load_start
    if not cached_tracks:
        raise ValueError("No tracks available in cache for the selected split/context.")

    max_velocity_steps = max(len(track['x_ctx']) - 1 for track in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))

    seed_cv_steps, seed_ctrv_steps = _load_branch_velocity_steps(diagnostics_dir)
    print_fn(
        "Cached hybrid objective with search space "
        f"cv_velocity_steps in [1, {max_velocity_steps}], "
        f"ctrv_velocity_steps in [1, {max_velocity_steps}], "
        f"rot_steps in [1, {max_velocity_steps}], "
        "rot_threshold in [0.0, 10.0] deg/min"
    )

    timing_stats = {'eval_calls': 0, 'eval_s': 0.0}
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20, n_ei_candidates=100, multivariate=True)
    study = optuna.create_study(
        study_name="hybrid_cv_ctrv_val_tpe",
        direction="minimize",
        sampler=sampler,
    )
    study.enqueue_trial({
        "cv_velocity_steps": seed_cv_steps,
        "ctrv_velocity_steps": seed_ctrv_steps,
        "rot_steps": seed_ctrv_steps,
        "rot_threshold": 1.0,
    })

    t_trials_start = time.perf_counter()
    study.optimize(
        make_hybrid_objective(
            cached_tracks,
            max_velocity_steps=max_velocity_steps,
            timing_stats=timing_stats,
        ),
        n_trials=HYBRID_N_TRIALS,
        n_jobs=1,
        callbacks=[RMSEEarlyStoppingCallback(patience=20, min_delta=1e-4)],
    )
    trials_s = time.perf_counter() - t_trials_start

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = "hybrid_cv_ctrv"
    trials_df["model_label"] = "Hybrid CV/CTRV"

    csv_path = diagnostics_dir / "tuning_hybrid_cv_ctrv_val.csv"
    trials_df.to_csv(csv_path, index=False)
    print_fn(f"\nTrial table saved to {csv_path}")

    best = study.best_trial
    best_cv_steps = int(best.params["cv_velocity_steps"])
    best_ctrv_steps = int(best.params["ctrv_velocity_steps"])
    best_rot_steps = int(best.params["rot_steps"])
    best_threshold = float(best.params["rot_threshold"])
    best_rmse = float(best.value)

    print_fn(f"\n{'='*50}")
    print_fn("  Model                 : Hybrid CV/CTRV")
    print_fn(f"  Seed CV velocity_steps: {seed_cv_steps}")
    print_fn(f"  Seed CTRV velocity_steps: {seed_ctrv_steps}")
    print_fn(f"  Best CV velocity_steps: {best_cv_steps}")
    print_fn(f"  Best CTRV velocity_steps: {best_ctrv_steps}")
    print_fn(f"  Best rot_steps        : {best_rot_steps}")
    print_fn(f"  Best rot_threshold    : {best_threshold:.6f} deg/min")
    print_fn(f"  Best val RMSE         : {best_rmse:.4f} m")
    print_fn(f"  Trials completed      : {len(trials_df)}")
    print_fn(f"{'='*50}")

    if ENABLE_TIMING_DIAGNOSTICS:
        eval_calls = max(1, int(timing_stats['eval_calls']))
        avg_eval_ms = (timing_stats['eval_s'] / eval_calls) * 1000.0
        overhead_s = max(0.0, trials_s - timing_stats['eval_s'])
        print_fn("\nTiming diagnostics:")
        print_fn(f"  Cache load time        : {load_s:.3f} s")
        print_fn(f"  Trial loop total       : {trials_s:.3f} s")
        print_fn(f"  Pure eval time         : {timing_stats['eval_s']:.3f} s")
        print_fn(f"  Avg eval per trial     : {avg_eval_ms:.1f} ms")
        print_fn(f"  Sampler/loop overhead  : {overhead_s:.3f} s")

    plot_path = diagnostics_dir / "tuning_hybrid_cv_ctrv_val_plot.png"
    plot_hybrid_results(
        df=trials_df,
        best_cv_steps=best_cv_steps,
        best_ctrv_steps=best_ctrv_steps,
        best_rot_steps=best_rot_steps,
        best_threshold=best_threshold,
        best_rmse=best_rmse,
        model_label="Hybrid CV/CTRV",
        output_path=plot_path,
        print_fn=print_fn,
    )

    return {
        "model_key": "hybrid_cv_ctrv",
        "model_label": "Hybrid CV/CTRV",
        "cv_velocity_steps": best_cv_steps,
        "ctrv_velocity_steps": best_ctrv_steps,
        "best_rot_steps": best_rot_steps,
        "best_rot_threshold": best_threshold,
        "best_val_rmse": best_rmse,
        "trials_csv": str(csv_path),
        "plot_png": str(plot_path),
    }, trials_df


def _run_model_job(result_queue, print_lock, model_key, model_label, model_cls, data_dir, diagnostics_dir):
    best_row, trials_df = run_optimization_for_model(
        model_key=model_key,
        model_label=model_label,
        model_cls=model_cls,
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
        print_lock=print_lock,
    )
    result_queue.put((best_row, trials_df))


def _run_hybrid_job(result_queue, print_lock, data_dir, diagnostics_dir):
    best_row, trials_df = run_hybrid_optimization(
        data_dir=data_dir,
        diagnostics_dir=diagnostics_dir,
        print_lock=print_lock,
    )
    result_queue.put((best_row, trials_df))


if __name__ == "__main__":
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

    if not standard_jobs and not RUN_HYBRID:
        print("No models selected. Set RUN_CV and/or RUN_CTRV and/or RUN_HYBRID to True.")
        raise SystemExit(0)

    best_rows = []
    all_trials = []

    result_queue = mp.Queue()
    print_lock = mp.Lock()
    processes = []

    if RUN_HYBRID and not standard_jobs:
        best_rows = _load_existing_best_rows(diagnostics_dir)
        print("Using existing tuning_best_params_val.csv for CV/CTRV hybrid seeding.")

    for model_key, model_label, model_cls in standard_jobs:
        p = mp.Process(
            target=_run_model_job,
            args=(
                result_queue,
                print_lock,
                model_key,
                model_label,
                model_cls,
                data_dir,
                diagnostics_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for _ in standard_jobs:
        best_row, trials_df = result_queue.get()
        best_rows.append(best_row)
        all_trials.append(trials_df)

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_path = diagnostics_dir / "tuning_best_params_val.csv"
        best_df.to_csv(best_path, index=False)
        print(f"\nBest-parameter summary saved to {best_path}")
        print(best_df.to_string(index=False))

    if RUN_HYBRID:
        hybrid_process = mp.Process(
            target=_run_hybrid_job,
            args=(
                result_queue,
                print_lock,
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
    print(best_df.to_string(index=False))

    all_trials_df = pd.concat(all_trials, ignore_index=True)
    all_trials_path = diagnostics_dir / "tuning_all_trials_val.csv"
    all_trials_df.to_csv(all_trials_path, index=False)
    print(f"All trials table saved to {all_trials_path}")

    print("\nNext step: use the saved best parameters for the final test evaluation.")
