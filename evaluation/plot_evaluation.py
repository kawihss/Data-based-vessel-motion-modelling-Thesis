

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from evaluation.evaluator import plot_horizon_error


# Config
EVAL_SPLIT = "test"
STEP_DURATION_S = 30  # seconds per prediction step

PLOT_CONSTANT_VELOCITY = True
PLOT_CTRV = True
PLOT_HYBRID = True

_ALL_MODEL_LABELS = {
    "constant_velocity": "Constant Velocity",
    "ctrv": "CTRV",
    "hybrid_cv_ctrv": "Hybrid CV/CTRV",
}
_MODEL_FLAGS = {
    "constant_velocity": PLOT_CONSTANT_VELOCITY,
    "ctrv": PLOT_CTRV,
    "hybrid_cv_ctrv": PLOT_HYBRID,
}
ALL_MODEL_LABELS = {k: v for k, v in _ALL_MODEL_LABELS.items() if _MODEL_FLAGS[k]}

METRICS_TO_PLOT = ["ADE", "FDE", "RMSE"]  # columns expected in per-month CSVs

project_root = Path(__file__).resolve().parent.parent
baseline_output_dir = project_root / "output" / "08_baseline_results"
diagnostics_dir = baseline_output_dir / "diagnostics"
plots_dir = baseline_output_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


# 1. ADE horizon plot
# 
def plot_ade_horizon():
    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False

    for key, label in ALL_MODEL_LABELS.items():
        ade_path = diagnostics_dir / f"{EVAL_SPLIT}_ade_per_step_{key}.csv"
        if not ade_path.exists():
            print(f"[horizon] No ADE_per_step file for '{label}', skipping.")
            continue
        ade_values = pd.read_csv(ade_path)["ade"].values
        plot_horizon_error({"ADE_per_step": ade_values}, label=label, ax=ax,
                           step_duration_s=STEP_DURATION_S)
        any_plotted = True

    if not any_plotted:
        print("[horizon] No ADE_per_step files found. Run run_evaluation.py first.")
        plt.close(fig)
        return

    ax.set_title(f"ADE over Prediction Horizon ({EVAL_SPLIT} split)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = plots_dir / f"{EVAL_SPLIT}_ade_horizon.png"
    fig.savefig(out, dpi=150)
    print(f"[horizon] Saved to {out}")
    plt.close(fig)


# 2. Monthly metrics plot
def _parse_month(val):
    # Parse strings like '2024-03' to a sortable (year, month) tuple.
    try:
        parts = str(val).split("-")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (9999, 99)


def plot_monthly_metrics():
    for metric in METRICS_TO_PLOT:
        fig, ax = plt.subplots(figsize=(11, 5))
        any_plotted = False

        for key, label in ALL_MODEL_LABELS.items():
            month_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{key}.csv"
            if not month_path.exists():
                print(f"[monthly/{metric}] No per-month file for '{label}', skipping.")
                continue
            df = pd.read_csv(month_path)
            if "month" not in df.columns or metric not in df.columns:
                print(f"[monthly/{metric}] '{label}' CSV missing 'month' or '{metric}' column, skipping.")
                continue

            df = df.dropna(subset=["month", metric])
            df = df.sort_values("month", key=lambda s: s.map(_parse_month))

            ax.plot(
                df["month"].astype(str),
                df[metric],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=label,
            )
            any_plotted = True

        if not any_plotted:
            print(f"[monthly/{metric}] No data found.")
            plt.close(fig)
            continue

        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel(f"{metric}  [m]", fontsize=11)
        ax.set_title(f"{metric} per Month ({EVAL_SPLIT} split)", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"{EVAL_SPLIT}_monthly_{metric.lower()}.png"
        fig.savefig(out, dpi=150)
        print(f"[monthly/{metric}] Saved to {out}")
        plt.close(fig)



def plot_results(df, best_steps, best_rmse, model_label, output_path):
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
    print(f"Plot saved to {output_path}")
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
    ax_mid.scatter(df["params_rot_steps"], df["value"], s=25, alpha=0.5, color="steelblue", label="All trials")
    ax_mid.scatter(
        [best_rot_steps], [best_rmse], color="crimson", marker="*", s=220, zorder=5,
        label=f"Best rot_steps={best_rot_steps}\nRMSE={best_rmse:.4f} m",
    )
    ax_mid.set_xlabel("rot_steps", fontsize=11)
    ax_mid.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax_mid.set_title("RMSE vs rot_steps (all trials)", fontsize=12)
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(fontsize=9)

    # Panel 3: RMSE vs rot_threshold for all trials (scatter), best point starred.
    ax_right.scatter(df["params_rot_threshold"], df["value"], s=25, alpha=0.5, color="steelblue", label="All trials")
    ax_right.scatter(
        [best_threshold], [best_rmse], color="crimson", marker="*", s=220, zorder=5,
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
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_tuning_results():
    tuning_diagnostics_dir = Path(__file__).resolve().parent / "diagnostics"
    if not tuning_diagnostics_dir.exists():
        print("[tuning] No tuning diagnostics directory found, skipping.")
        return

    best_params_path = tuning_diagnostics_dir / "tuning_best_params_val.csv"
    if not best_params_path.exists():
        print("[tuning] No tuning_best_params_val.csv found, skipping.")
        return

    best_params = pd.read_csv(best_params_path)

    for _, row in best_params.iterrows():
        model_key = str(row.get("model_key", "")).strip().lower()
        model_label = str(row.get("model_label", model_key))

        if model_key == "hybrid_cv_ctrv":
            csv_path = tuning_diagnostics_dir / "tuning_hybrid_cv_ctrv_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping Hybrid.")
                continue
            df = pd.read_csv(csv_path)
            plot_path = plots_dir / "tuning_hybrid_cv_ctrv_val_plot.png"
            plot_hybrid_results(
                df=df,
                best_cv_steps=int(row["cv_velocity_steps"]),
                best_ctrv_steps=int(row["ctrv_velocity_steps"]),
                best_rot_steps=int(row["best_rot_steps"]),
                best_threshold=float(row["best_rot_threshold"]),
                best_rmse=float(row["best_val_rmse"]),
                model_label=model_label,
                output_path=plot_path,
            )
        else:
            csv_path = tuning_diagnostics_dir / f"tuning_{model_key}_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping {model_label}.")
                continue
            df = pd.read_csv(csv_path)
            plot_path = plots_dir / f"tuning_{model_key}_val_plot.png"
            plot_results(
                df=df,
                best_steps=int(row["best_velocity_steps"]),
                best_rmse=float(row["best_val_rmse"]),
                model_label=model_label,
                output_path=plot_path,
            )



if __name__ == "__main__":
    print(f"Reading diagnostics from: {diagnostics_dir}")
    print(f"Saving plots to: {plots_dir}\n")

    plot_ade_horizon()
    plot_monthly_metrics()
    plot_tuning_results()

    print("\nDone.")
