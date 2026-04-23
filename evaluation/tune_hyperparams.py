import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel
from evaluation.evaluator import load_tracks_cached_numpy, evaluate_model_cached_numpy

CONTEXT_FILTER = None   # None = all contexts; or e.g. 'lock'
RUN_CV = True
RUN_CTRV = True
N_TRIALS = 20
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


if __name__ == "__main__":
    import multiprocessing as mp

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "output" / "07_parquet"
    diagnostics_dir = project_root / "evaluation" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tuning velocity_steps with {N_TRIALS} trial(s) per model")
    print("Validation tracks are cached once in RAM per model process\n")

    jobs = []
    if RUN_CV:
        jobs.append(("cv", "Constant Velocity", ConstantVelocityModel))
    if RUN_CTRV:
        jobs.append(("ctrv", "CTRV", ConstantTurnRateVelocityModel))

    if not jobs:
        print("No models selected. Set RUN_CV and/or RUN_CTRV to True.")
        raise SystemExit(0)

    best_rows = []
    all_trials = []

    result_queue = mp.Queue()
    print_lock = mp.Lock()
    processes = []

    for model_key, model_label, model_cls in jobs:
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

    for _ in jobs:
        best_row, trials_df = result_queue.get()
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

    print("\nNext step: copy best_velocity_steps values into evaluation/run_evaluation.py for split='test'.")
