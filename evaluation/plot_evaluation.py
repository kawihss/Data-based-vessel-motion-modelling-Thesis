"""
plot_evaluation.py
==================
Standalone plotting script for evaluation results.
Reads CSVs saved by run_evaluation.py — does NOT rerun any model.

Plots produced
--------------
1. ADE horizon plot  — ADE vs prediction horizon for all models.
2. Monthly metrics   — ADE / FDE / RMSE per calendar month for each model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from evaluation.evaluator import plot_horizon_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 1. ADE horizon plot
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Monthly metrics plot
# ---------------------------------------------------------------------------
def _parse_month(val):
    """Return a (year, month) tuple for sorting from strings like '2024-03'."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Reading diagnostics from: {diagnostics_dir}")
    print(f"Saving plots to: {plots_dir}\n")

    plot_ade_horizon()
    plot_monthly_metrics()

    print("\nDone.")
