import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.kinematic import ConstantVelocityModel, ConstantTurnRateVelocityModel
from evaluation.evaluator import run_evaluation

N_GRID_POINTS = 20
GRID = list(np.linspace(0.05, 1.0, N_GRID_POINTS).round(4))
CONTEXT_FILTER = None   # None = all contexts; or e.g. 'lock'
RUN_CV = True
RUN_CTRV = True


def make_objective(model_cls, data_dir):
    def objective(trial):
        vf = trial.suggest_categorical("velocity_fraction", GRID)
        model = model_cls(velocity_fraction=vf)
        metrics = run_evaluation(
            model,
            data_dir,
            split="val",
            context_filter=CONTEXT_FILTER,
            export_predictions=False,
        )
        return metrics["RMSE"]
    return objective


def plot_results(df, best_vf, best_rmse, model_label, output_path):
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
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def run_grid_search_for_model(model_key, model_label, model_cls, data_dir, diagnostics_dir):
    print(f"\n=== {model_label} ===")
    sampler = optuna.samplers.GridSampler({"velocity_fraction": GRID})
    study = optuna.create_study(
        study_name=f"{model_key}_val_gridsearch",
        direction="minimize",
        sampler=sampler,
    )

    total = len(GRID)
    for i in range(1, total + 1):
        print(f"  Trial {i:>2}/{total} ...", end="", flush=True)
        study.optimize(make_objective(model_cls, data_dir), n_trials=1)
        last = study.trials[-1]
        trial_vf = float(last.params["velocity_fraction"])
        print(f"  velocity_fraction={trial_vf:.4f}  RMSE={last.value:.4f} m")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df["model_key"] = model_key
    trials_df["model_label"] = model_label

    csv_path = diagnostics_dir / f"tuning_{model_key}_val.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"\nTrial table saved to {csv_path}")

    rmse_counts = trials_df["value"].round(6).value_counts()
    repeated = rmse_counts[rmse_counts > 1]
    if not repeated.empty:
        print("\nRepeated RMSE values (rounded to 6 dp):")
        for rmse_value, count in repeated.items():
            print(f"  RMSE={rmse_value:.6f} appears {int(count)}x")

    best = study.best_trial
    best_vf = float(best.params["velocity_fraction"])
    best_rmse = float(best.value)

    print(f"\n{'='*50}")
    print(f"  Model                 : {model_label}")
    print(f"  Best velocity_fraction: {best_vf:.4f}")
    print(f"  Best val RMSE         : {best_rmse:.4f} m")
    print(f"{'='*50}")

    plot_path = diagnostics_dir / f"tuning_{model_key}_val_plot.png"
    plot_results(trials_df, best_vf, best_rmse, model_label, plot_path)

    return {
        "model_key": model_key,
        "model_label": model_label,
        "best_velocity_fraction": best_vf,
        "best_val_rmse": best_rmse,
        "trials_csv": str(csv_path),
        "plot_png": str(plot_path),
    }, trials_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "output" / "05_normalized"
    diagnostics_dir = project_root / "evaluation" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid search over {N_GRID_POINTS} velocity_fraction values: {GRID[0]} … {GRID[-1]}")
    print(f"Each trial = one full val-split evaluation  (~2–3 min each)\n")

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

    for model_key, model_label, model_cls in jobs:
        best_row, trials_df = run_grid_search_for_model(
            model_key=model_key,
            model_label=model_label,
            model_cls=model_cls,
            data_dir=data_dir,
            diagnostics_dir=diagnostics_dir,
        )
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

    print("\nNext step: copy best_velocity_fraction values into evaluation/run_evaluation.py for split='test'.")
