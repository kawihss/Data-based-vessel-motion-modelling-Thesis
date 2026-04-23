import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel
from evaluation.evaluator import load_tracks_cached, evaluate_model_cached

CONTEXT_FILTER = None   # None = all contexts; or e.g. 'lock'
RUN_CV = True
RUN_CTRV = True
N_TRIALS = 20


def _safe_print(print_lock, *args, **kwargs):
    if print_lock is None:
        print(*args, **kwargs)
        return
    with print_lock:
        print(*args, **kwargs)


def make_objective(model_cls, cached_tracks, max_velocity_steps):
    def objective(trial):
        velocity_steps = trial.suggest_int("velocity_steps", 1, max_velocity_steps)
        model = model_cls(velocity_steps=velocity_steps)
        metrics = evaluate_model_cached(model, cached_tracks)
        return metrics["RMSE"]
    return objective


def plot_results(df, best_vf, best_rmse, model_label, output_path, print_fn=print):
    df_sorted = df.sort_values("params_velocity_fraction")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        df_sorted["params_velocity_fraction"],
        df_sorted["value"],
        marker="o",
        linewidth=1.8,
        markersize=5,
        color="steelblue",
        label="Val RMSE",
    )
    ax.axvline(best_vf, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Best k={best_vf:.4f}  (RMSE={best_rmse:.4f} m)")
    ax.annotate(
        f"k={best_vf:.4f}\nRMSE={best_rmse:.4f} m",
        xy=(best_vf, best_rmse),
        xytext=(best_vf + 0.05, best_rmse + (df_sorted["value"].max() - df_sorted["value"].min()) * 0.08),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        fontsize=9,
        color="crimson",
    )
    ax.set_xlabel("velocity_fraction  (k – fraction of context steps averaged)", fontsize=11)
    ax.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax.set_title(f"{model_label} - Hyperparameter Grid Search\nVal-split RMSE vs velocity_fraction", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print_fn(f"Plot saved to {output_path}")
    plt.close(fig)


def run_optimization_for_model(model_key, model_label, model_cls, data_dir, diagnostics_dir, print_lock=None):
    print_fn = lambda *args, **kwargs: _safe_print(print_lock, *args, **kwargs)

    print_fn(f"\n=== {model_label} ===")
    cached_tracks = load_tracks_cached(data_dir, split='val', context_filter=CONTEXT_FILTER)
    if not cached_tracks:
        raise ValueError("No tracks available in cache for the selected split/context.")
    max_velocity_steps = max(len(context_df) - 1 for context_df, _ in cached_tracks)
    max_velocity_steps = max(1, int(max_velocity_steps))
    print_fn(f"Cached objective with velocity_steps in [1, {max_velocity_steps}]")

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5, n_ei_candidates=100)
    study = optuna.create_study(
        study_name=f"{model_key}_val_gridsearch",
        direction="minimize",
        sampler=sampler,
    )

    total = N_TRIALS
    for i in range(1, total + 1):
        study.optimize(make_objective(model_cls, cached_tracks, max_velocity_steps), n_trials=1, n_jobs=1)
        last = study.trials[-1]
        trial_steps = int(last.params["velocity_steps"])
        print_fn(
            f"[{model_key.upper()}] Trial {i:>2}/{total}  "
            f"velocity_steps={trial_steps}  RMSE={last.value:.4f} m"
        )

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

    plot_path = diagnostics_dir / f"tuning_{model_key}_val_plot.png"
    plot_results(
        df=trials_df.rename(columns={"params_velocity_steps": "params_velocity_fraction"}),
        best_vf=float(best_steps),
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
    data_dir = project_root / "output" / "05_normalized"
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
