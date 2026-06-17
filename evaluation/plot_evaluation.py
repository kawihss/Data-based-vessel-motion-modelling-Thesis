import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from evaluation.evaluator import plot_horizon_error
from evaluation.runtime_config import load_runtime_config, resolve_run_paths


#  loads evaluation/tuning CSV outputs from the selected run directory
# and generates all publication-style plots for model comparison:
# - Evaluation quality plots (horizon, monthly, context, uncertainty)
# - Feature importance plots (channel and timestep)
# - Tuning diagnostics (search-space and convergence).. (but you should use optuna plots when available, this was implemented before i knew about them)
# - Pairwise RMSE distribution plots (violin)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_runtime_config(PROJECT_ROOT)

# Config
EVAL_SPLIT = str(CONFIG["data"]["split"])
STEP_DURATION_S = int(CONFIG["plotting"]["step_duration_s"])

PLOT_FLAGS = CONFIG["plotting"]["models"]
PLOT_CONSTANT_VELOCITY = bool(PLOT_FLAGS["constant_velocity"])
PLOT_CTRV = bool(PLOT_FLAGS["ctrv"])
PLOT_CTRV_ARC = bool(PLOT_FLAGS["ctrv_arc"])
PLOT_HYBRID = bool(PLOT_FLAGS["hybrid_cv_ctrv"])
PLOT_KALMAN = bool(PLOT_FLAGS["kalman"])
PLOT_CTRV_EKF = bool(PLOT_FLAGS["ctrv_ekf"])
PLOT_TIREX_LSTM = bool(PLOT_FLAGS["tirex_lstm"])
PLOT_CHRONOS2_ZERO_SHOT = bool(PLOT_FLAGS.get("chronos2_zero_shot", False))
PLOT_MINIMAL_LSTM = bool(PLOT_FLAGS.get("minimal_lstm", False))
PLOT_MINIMAL_LSTM_DOMAIN = bool(PLOT_FLAGS.get("minimal_lstm_domain", PLOT_FLAGS.get("embedded_lstm", False)))

_ALL_MODEL_LABELS = {
    "constant_velocity": "Constant Velocity",
    "ctrv": "CTRV",
    "ctrv_arc": "CTRV Arc",
    "hybrid_cv_ctrv": "Hybrid CV/CTRV",
    "kalman": "Kalman",
    "ctrv_ekf": "CTRV EKF",
    "tirex_lstm": "TiRex LSTM",
    "chronos2_zero_shot": "Chronos-2 Zero-Shot",
    "minimal_lstm": "Minimal LSTM",
    "minimal_lstm_domain": "OHE LSTM",
}
_MODEL_FLAGS = {
    "constant_velocity": PLOT_CONSTANT_VELOCITY,
    "ctrv": PLOT_CTRV,
    "ctrv_arc": PLOT_CTRV_ARC,
    "hybrid_cv_ctrv": PLOT_HYBRID,
    "kalman": PLOT_KALMAN,
    "ctrv_ekf": PLOT_CTRV_EKF,
    "tirex_lstm": PLOT_TIREX_LSTM,
    "chronos2_zero_shot": PLOT_CHRONOS2_ZERO_SHOT,
    "minimal_lstm": PLOT_MINIMAL_LSTM,
    "minimal_lstm_domain": PLOT_MINIMAL_LSTM_DOMAIN,
}
ALL_MODEL_LABELS = {k: v for k, v in _ALL_MODEL_LABELS.items() if _MODEL_FLAGS[k]}

METRICS_TO_PLOT = ["ADE", "FDE", "RMSE"]  # expected columns in per-month CSVs
CONTEXTS_TO_PLOT = ["harbour", "river", "channel", "lock"]
CONTEXT_METRICS_TO_PLOT = ["ADE", "FDE", "RMSE"]
UNCERTAINTY_METRICS_TO_PLOT = ["MIW", "Coverage", "Winkler80"]

run_paths = resolve_run_paths(PROJECT_ROOT, CONFIG, create=False)
selected_run_dir = run_paths["run_dir"]
if not selected_run_dir.exists():
    raise FileNotFoundError(
        f"Configured run directory not found: {selected_run_dir}. "
        "Set run.name in configs/evaluation.yaml to an existing run or execute run_evaluation.py first."
    )

diagnostics_dir = selected_run_dir / "diagnostics"
plots_dir = selected_run_dir / "plots"
tuning_diagnostics_dir = selected_run_dir / "tuning"
plots_dir.mkdir(parents=True, exist_ok=True)


# ---------- Evaluation plot group ----------
# 1) ADE horizon plot
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
def _parse_month(val): # greyed out i vscode, but required
    # Parse strings like '2024-03' to a sortable (year, month) tuple.
    try:
        parts = str(val).split("-")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (9999, 99)


def plot_monthly_metrics():
    # Line plots for ADE/FDE/RMSE over months per enabled model
    for metric in METRICS_TO_PLOT:
        fig, ax = plt.subplots(figsize=(11, 5))
        any_plotted = False

        for key, label in ALL_MODEL_LABELS.items():
            month_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_month_{key}.csv"
            if not month_path.exists():
                continue
            df = pd.read_csv(month_path)
            if "month" not in df.columns or metric not in df.columns:
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
                alpha=0.7,
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


def _plot_context_metric_group(metrics, output_prefix):
    # Shared grouped-bar plotting logic for context and uncertainty metrics
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(11, 5))
        any_plotted = False
        x_positions = np.arange(len(CONTEXTS_TO_PLOT), dtype=float)
        width = 0.8 / max(len(ALL_MODEL_LABELS), 1)

        for idx, (key, label) in enumerate(ALL_MODEL_LABELS.items()):
            context_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_context_{key}.csv"
            if not context_path.exists():
                continue
            df = pd.read_csv(context_path)
            if "context" not in df.columns or metric not in df.columns:
                continue

            aligned = (
                df.set_index("context")
                .reindex(CONTEXTS_TO_PLOT)[metric]
                .astype(float)
            )
            offset = (idx - (len(ALL_MODEL_LABELS) - 1) / 2.0) * width
            ax.bar(x_positions + offset, aligned.values, width=width, label=label)
            any_plotted = True

        if not any_plotted:
            print(f"[context/{metric}] No data found.")
            plt.close(fig)
            continue

        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONTEXTS_TO_PLOT)
        ax.set_xlabel("Context", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} by Geographic Context ({EVAL_SPLIT} split)", fontsize=13)
        if metric == "Coverage":
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plots_dir / f"{EVAL_SPLIT}_{output_prefix}_{metric.lower()}.png"
        fig.savefig(out, dpi=150)
        print(f"[context/{metric}] Saved to {out}")
        plt.close(fig)


def plot_context_metrics():
    # Context-wise ADE/FDE/RMSE.
    _plot_context_metric_group(CONTEXT_METRICS_TO_PLOT, output_prefix="context")


def plot_uncertainty_metrics():
    # Context-wise MIW/Coverage/Winkler80.
    _plot_context_metric_group(UNCERTAINTY_METRICS_TO_PLOT, output_prefix="uncertainty")


def plot_channel_importance():
    # normalized covariate importance per context, shown as 2x2 panels of bar charts per model
    for key, label in ALL_MODEL_LABELS.items():
        importance_path = diagnostics_dir / f"{EVAL_SPLIT}_channel_importance_{key}.csv"
        if not importance_path.exists():
            continue

        df = pd.read_csv(importance_path)
        if df.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
        axes = axes.flatten()
        any_plotted = False

        for axis, context_label in zip(axes, CONTEXTS_TO_PLOT):
            ctx_df = df[df["context"] == context_label].copy()
            if ctx_df.empty:
                axis.set_visible(False)
                continue

            ctx_df = ctx_df.sort_values("importance", ascending=False)
            axis.bar(ctx_df["covariate"], ctx_df["importance"], color="steelblue")
            axis.set_title(context_label)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.tick_params(axis="x", rotation=45)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        fig.suptitle(f"Channel Importance by Context ({label})", fontsize=13)
        fig.tight_layout()
        out = plots_dir / f"{EVAL_SPLIT}_channel_importance_{key}.png"
        fig.savefig(out, dpi=150)
        print(f"[channel-importance] Saved to {out}")
        plt.close(fig)


def plot_timestep_importance():
    # 2x2 context panels showing normalized context-timestep importance
    for key, label in ALL_MODEL_LABELS.items():
        importance_path = diagnostics_dir / f"{EVAL_SPLIT}_timestep_importance_{key}.csv"
        if not importance_path.exists():
            continue

        df = pd.read_csv(importance_path)
        if df.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
        axes = axes.flatten()
        any_plotted = False

        for axis, context_label in zip(axes, CONTEXTS_TO_PLOT):
            ctx_df = df[df["context"] == context_label].copy()
            if ctx_df.empty:
                axis.set_visible(False)
                continue

            ctx_df = ctx_df.sort_values("timestep")
            ctx_df = ctx_df[ctx_df["timestep"] > 1]
            axis.plot(ctx_df["timestep"], ctx_df["importance"], marker="o", color="darkorange")
            axis.set_title(context_label)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("Context Timestep")
            axis.grid(True, alpha=0.3)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        axes[0].set_ylabel("Normalized Importance")
        axes[2].set_ylabel("Normalized Importance")
        fig.suptitle(f"Timestep Importance by Context ({label})", fontsize=13)
        fig.tight_layout()
        out = plots_dir / f"{EVAL_SPLIT}_timestep_importance_{key}.png"
        fig.savefig(out, dpi=150)
        print(f"[timestep-importance] Saved to {out}")
        plt.close(fig)



# ---------- Tuning diagnostics plot group ----------
#prefer optuna plots for tuning diagnostics when available
def plot_results(df, best_steps, best_rmse, model_label, output_path):
    # Generic 1D velocity_steps tuning curve (RMSE vs steps)
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
        alpha=0.7,
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
    # Hybrid-specific diagnostics: pair heatmap + two sensitivity scatters. prefer optuna plots
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
    ax_mid.scatter(df["params_rot_steps"], df["value"], s=25, alpha=0.7, color="steelblue", label="All trials")
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
    ax_right.scatter(df["params_rot_threshold"], df["value"], s=25, alpha=0.7, color="steelblue", label="All trials")
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


def plot_kalman_results(df, best_params, best_rmse, model_label, output_path):
    # Scatter diagnostics per Kalman parameter (log-scale x-axes).
    param_cols = {
        "params_q_pos": "q_pos (process noise, position)",
        "params_q_vel": "q_vel (process noise, velocity)",
        "params_r_pos": "r_pos (measurement noise)",
        "params_p0_pos": "p0_pos (initial covariance, position)",
        "params_p0_vel": "p0_vel (initial covariance, velocity)",
    }
    available = {col: lbl for col, lbl in param_cols.items() if col in df.columns}
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (col, xlabel) in zip(axes, available.items()):
        ax.scatter(df[col], df["value"], s=20, alpha=0.7, color="steelblue", label="All trials")
        best_val = best_params.get(col.replace("params_", "best_"))
        if best_val is not None:
            ax.scatter([best_val], [best_rmse], color="crimson", marker="*", s=220, zorder=5,
                       label=f"Best: {best_val:.3g}\nRMSE={best_rmse:.4f} m")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Validation RMSE  [m]" if ax is axes[0] else "", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_label} - Hyperparameter Optimization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_ctrv_ekf_results(df, best_params, best_rmse, model_label, output_path):
    # Scatter diagnostics per CTRV-EKF parameter (log-scale x-axes).
    param_cols = {
        "params_q_pos": "q_pos (process noise, position)",
        "params_q_vel": "q_vel (process noise, velocity)",
        "params_q_rot": "q_rot (process noise, turn rate)",
        "params_r_pos": "r_pos (measurement noise)",
        "params_p0_pos": "p0_pos (initial covariance, position)",
        "params_p0_vel": "p0_vel (initial covariance, velocity)",
        "params_p0_rot": "p0_rot (initial covariance, turn rate)",
    }
    available = {col: lbl for col, lbl in param_cols.items() if col in df.columns}
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (col, xlabel) in zip(axes, available.items()):
        ax.scatter(df[col], df["value"], s=20, alpha=0.7, color="steelblue", label="All trials")
        best_val = best_params.get(col.replace("params_", "best_"))
        if best_val is not None:
            ax.scatter([best_val], [best_rmse], color="crimson", marker="*", s=220, zorder=5,
                       label=f"Best: {best_val:.3g}\nRMSE={best_rmse:.4f} m")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Validation RMSE  [m]" if ax is axes[0] else "", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_label} - Hyperparameter Optimization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_convergence(df, model_label, output_path):
    # Optuna convergence: trial values and cumulative best value.
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"].copy()
    if "number" not in df.columns or "value" not in df.columns or df.empty:
        print(f"[convergence] Missing or empty trial data for {model_label}, skipping.")
        return

    df = df.sort_values("number").reset_index(drop=True)
    df["best_so_far"] = df["value"].cummin()
    best_idx = int(df["value"].idxmin())
    best_trial = int(df.loc[best_idx, "number"])
    best_rmse = float(df.loc[best_idx, "value"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["number"], df["value"], s=18, color="steelblue", alpha=0.7, zorder=3, label="Trial RMSE")
    ax.step(df["number"], df["best_so_far"], where="post", color="crimson", linewidth=2, label="Best so far")
    ax.axvline(best_trial, color="green", linestyle="--", linewidth=1.2,
               label=f"Best trial ({best_trial}, RMSE={best_rmse:.4f} m)")

    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Validation RMSE  [m]", fontsize=11)
    ax.set_title(f"{model_label} - Optuna Convergence", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    x_min = int(df["number"].min()) - 1
    x_max = int(df["number"].max()) + 1
    ax.set_xlim(x_min, x_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[convergence] Saved to {output_path}")
    plt.close(fig)


def plot_minimal_lstm_results(df, best_params, best_rmse, model_label, output_path):
    # Minimal-LSTM hyperparameter sensitivity scatter panels.
    param_cols = {
        "params_hidden_size": "hidden_size (units)",
        "params_num_layers": "num_layers",
        "params_dropout": "dropout",
        "params_learning_rate": "learning_rate",
        "params_batch_size": "batch_size",
    }
    available = {col: lbl for col, lbl in param_cols.items() if col in df.columns}
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (col, xlabel) in zip(axes, available.items()):
        ax.scatter(df[col], df["value"], s=20, alpha=0.7, color="steelblue", label="All trials")
        best_val = best_params.get(col.replace("params_", "best_"))
        if best_val is not None:
            ax.scatter([best_val], [best_rmse], color="crimson", marker="*", s=220, zorder=5,
                       label=f"Best: {best_val}\nRMSE={best_rmse:.4f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Validation Loss" if ax is axes[0] else "", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_label} - Hyperparameter Optimization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_tuning_results():
    # Dispatch to model-specific tuning plot builders
    if not tuning_diagnostics_dir.exists():
        print("[tuning] No tuning diagnostics directory found, skipping.")
        return

    best_params_path = tuning_diagnostics_dir / "tuning_best_params_val.csv"
    if not best_params_path.exists() or best_params_path.stat().st_size == 0:
        print("[tuning] No tuning_best_params_val.csv found, skipping.")
        return

    best_params_df = pd.read_csv(best_params_path)
    for _, row in best_params_df.iterrows():
        model_key = str(row.get("model_key", "")).strip().lower()
        model_label = str(row.get("model_label", model_key))

        if model_key == "hybrid_cv_ctrv":
            csv_path = tuning_diagnostics_dir / "tuning_hybrid_cv_ctrv_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping Hybrid.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / "tuning_hybrid_cv_ctrv_convergence.png",
            )
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
        elif model_key == "kalman":
            csv_path = tuning_diagnostics_dir / "tuning_kalman_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping Kalman.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / "tuning_kalman_convergence.png",
            )
            plot_path = plots_dir / "tuning_kalman_val_plot.png"
            best_params = {
                "best_q_pos": float(row["best_q_pos"]),
                "best_q_vel": float(row["best_q_vel"]),
                "best_r_pos": float(row["best_r_pos"]),
                "best_p0_pos": float(row["best_p0_pos"]),
                "best_p0_vel": float(row["best_p0_vel"]),
            }
            plot_kalman_results(
                df=df,
                best_params=best_params,
                best_rmse=float(row["best_val_rmse"]),
                model_label=model_label,
                output_path=plot_path,
            )
        elif model_key == "ctrv_ekf":
            csv_path = tuning_diagnostics_dir / "tuning_ctrv_ekf_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping CTRV EKF.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / "tuning_ctrv_ekf_convergence.png",
            )
            plot_path = plots_dir / "tuning_ctrv_ekf_val_plot.png"
            best_params = {
                "best_q_pos": float(row["best_q_pos"]),
                "best_q_vel": float(row["best_q_vel"]),
                "best_q_rot": float(row["best_q_rot"]),
                "best_r_pos": float(row["best_r_pos"]),
                "best_p0_pos": float(row["best_p0_pos"]),
                "best_p0_vel": float(row["best_p0_vel"]),
                "best_p0_rot": float(row["best_p0_rot"]),
            }
            plot_ctrv_ekf_results(
                df=df,
                best_params=best_params,
                best_rmse=float(row["best_val_rmse"]),
                model_label=model_label,
                output_path=plot_path,
            )
        elif model_key == "minimal_lstm":
            csv_path = tuning_diagnostics_dir / "tuning_minimal_lstm_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping Minimal LSTM.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / "tuning_minimal_lstm_convergence.png",
            )
            plot_path = plots_dir / "tuning_minimal_lstm_val_plot.png"
            best_params = {
                "best_hidden_size": int(row["best_hidden_size"]),
                "best_num_layers": int(row["best_num_layers"]),
                "best_dropout": float(row["best_dropout"]),
                "best_learning_rate": float(row["best_learning_rate"]),
                "best_batch_size": int(row["best_batch_size"]),
            }
            plot_minimal_lstm_results(
                df=df,
                best_params=best_params,
                best_rmse=float(row["best_val_loss"]),
                model_label=model_label,
                output_path=plot_path,
            )
        elif model_key == "minimal_lstm_domain":
            csv_path = tuning_diagnostics_dir / "tuning_minimal_lstm_domain_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping Minimal LSTM Domain.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / "tuning_minimal_lstm_domain_convergence.png",
            )
            plot_path = plots_dir / "tuning_minimal_lstm_domain_val_plot.png"
            best_params = {
                "best_hidden_size": int(row["best_hidden_size"]),
                "best_num_layers": int(row["best_num_layers"]),
                "best_dropout": float(row["best_dropout"]),
                "best_learning_rate": float(row["best_learning_rate"]),
                "best_batch_size": int(row["best_batch_size"]),
            }
            plot_minimal_lstm_results(
                df=df,
                best_params=best_params,
                best_rmse=float(row["best_val_loss"]),
                model_label=model_label,
                output_path=plot_path,
            )
        else:
            csv_path = tuning_diagnostics_dir / f"tuning_{model_key}_val.csv"
            if not csv_path.exists():
                print(f"[tuning] {csv_path} not found, skipping {model_label}.")
                continue
            df = pd.read_csv(csv_path)
            plot_convergence(
                df=df,
                model_label=model_label,
                output_path=plots_dir / f"tuning_{model_key}_convergence.png",
            )
            plot_path = plots_dir / f"tuning_{model_key}_val_plot.png"
            if "best_velocity_steps" not in row.index:
                print(f"[tuning] best_velocity_steps missing for {model_label} ({model_key}), skipping generic tuning plot.")
                continue
            plot_results(
                df=df,
                best_steps=int(row["best_velocity_steps"]),
                best_rmse=float(row["best_val_rmse"]),
                model_label=model_label,
                output_path=plot_path,
            )



# ---------- Pairwise RMSE distribution (violin) plot group ----------
def _load_per_file_rmse(model_keys_labels):
    all_data = []
    for key, label in model_keys_labels:
        per_file_path = diagnostics_dir / f"{EVAL_SPLIT}_metrics_per_file_{key}.csv"
        if not per_file_path.exists():
            print(f"[violin] No per-file metrics for '{label}', skipping.")
            continue
        df = pd.read_csv(per_file_path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else None


def _save_violin(fig, ax, title, xlabel, ylabel, out_path, xrot=0):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=xrot, ha="right" if xrot else "center", fontsize=9)
    ax.legend(title="Modell")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[violin] Saved to {out_path}")
    plt.close(fig)


def plot_violin_cv_vs_kalman():
    pairs = [("constant_velocity", "CV"), ("kalman", "Kalman")]
    plot_df = _load_per_file_rmse(pairs)
    if plot_df is None:
        return
    for x_col, xlabel, xrot, suffix in [
        ("year", "Jahr", 0, "per_year"),
        ("month", "Monat", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            split=True, inner="quartile", cut=0, scale="width", ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: CV vs Kalman ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            plots_dir / f"{EVAL_SPLIT}_violin_cv_vs_kalman_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_ctrv_vs_ekf():
    pairs = [("ctrv", "CTRV"), ("ctrv_ekf", "CTRV EKF")]
    plot_df = _load_per_file_rmse(pairs)
    if plot_df is None:
        return
    for x_col, xlabel, xrot, suffix in [
        ("year", "Jahr", 0, "per_year"),
        ("month", "Monat", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            split=True, inner="quartile", cut=0, scale="width", ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: CTRV vs CTRV EKF ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            plots_dir / f"{EVAL_SPLIT}_violin_ctrv_vs_ekf_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_chronos_vs_tirex():
    if not (PLOT_CHRONOS2_ZERO_SHOT and PLOT_TIREX_LSTM):
        return
    pairs = [("chronos2_zero_shot", "Chronos-2 Zero-Shot"), ("tirex_lstm", "TiRex LSTM")]
    plot_df = _load_per_file_rmse(pairs)
    if plot_df is None:
        return
    for x_col, xlabel, xrot, suffix in [
        ("year", "Jahr", 0, "per_year"),
        ("month", "Monat", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            split=True, inner="quartile", cut=0, scale="width", ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: Chronos-2 vs TiRex LSTM ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            plots_dir / f"{EVAL_SPLIT}_violin_chronos_vs_tirex_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_lstm_vs_lstm_domain():
    runs_base = PROJECT_ROOT / "output" / "08_baseline_results" / "runs"
    sources = [
        (runs_base / "lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm.csv", "LSTM"),
        (runs_base / "OHE_lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm_domain.csv", "LSTM (Domain)"),
    ]

    all_data = []
    for path, label in sources:
        if not path.exists():
            print(f"[violin lstm vs domain] File not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin lstm vs domain] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)

    if not all_data:
        print("[violin lstm vs domain] No data found, skipping.")
        return

    plot_df = pd.concat(all_data, ignore_index=True)
    out_dir = PROJECT_ROOT / "output" / "08_baseline_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col, xlabel, xrot, suffix in [
        ("year", "Year", 0, "per_year"),
        ("month", "Month", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            split=True, inner="quartile", cut=0, scale="width", ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: LSTM vs LSTM (Domain) ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            out_dir / f"{EVAL_SPLIT}_violin_lstm_vs_lstm_domain_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_lstm_vs_nohpo_lstm_domain():
    runs_base = PROJECT_ROOT / "output" / "08_baseline_results" / "runs"
    sources = [
        (runs_base / "lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm.csv", "LSTM"),
        (runs_base / "noHPO_OHE_lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm_domain.csv", "LSTM + Domain *"),
    ]

    all_data = []
    for path, label in sources:
        if not path.exists():
            print(f"[violin lstm vs domain] File not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin lstm vs domain] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)

    if not all_data:
        print("[violin lstm vs domain] No data found, skipping.")
        return

    plot_df = pd.concat(all_data, ignore_index=True)
    out_dir = PROJECT_ROOT / "output" / "08_baseline_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col, xlabel, xrot, suffix in [
        ("year", "Year", 0, "per_year"),
        ("month", "Month", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            split=True, inner="quartile", cut=0, scale="width", ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: LSTM vs LSTM + Domain * ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            out_dir / f"{EVAL_SPLIT}_violin_lstm_vs_nohpo_lstm_domain_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_cv_ekf_chronos_lstm():
    # Violin plot comparing CV, EKF, Chronos and LSTM across all models
    runs_base = PROJECT_ROOT / "output" / "08_baseline_results" / "runs"
    sources = [
        (runs_base / "baseline_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_constant_velocity.csv", "CV"),
        (runs_base / "baseline_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_ctrv_ekf.csv", "EKF"),
        (runs_base / "foundation_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_chronos2_zero_shot.csv", "Chronos"),
        (runs_base / "lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm.csv", "LSTM"),
    ]

    all_data = []
    for path, label in sources:
        if not path.exists():
            print(f"[violin 4-models] File not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin 4-models] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)

    if not all_data:
        print("[violin 4-models] No data found, skipping.")
        return

    plot_df = pd.concat(all_data, ignore_index=True)
    out_dir = PROJECT_ROOT / "output" / "08_baseline_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col, xlabel, xrot, suffix in [
        ("year", "Year", 0, "per_year"),
        ("month", "Month", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            inner="quartile", cut=0, ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: CV vs EKF vs Chronos vs LSTM ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            out_dir / f"{EVAL_SPLIT}_violin_cv_ekf_chronos_lstm_{suffix}.png",
            xrot=xrot,
        )


def plot_violin_cv_ekf_tirex_lstm():
    # Violin plot comparing CV, EKF, TiRex and LSTM across all models
    runs_base = PROJECT_ROOT / "output" / "08_baseline_results" / "runs"
    sources = [
        (runs_base / "baseline_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_constant_velocity.csv", "CV"),
        (runs_base / "baseline_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_ctrv_ekf.csv", "EKF"),
        (runs_base / "foundation_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_tirex_lstm.csv", "TiRex"),
        (runs_base / "lstm_full_full_100" / "diagnostics" / f"{EVAL_SPLIT}_metrics_per_file_minimal_lstm.csv", "LSTM"),
    ]

    all_data = []
    for path, label in sources:
        if not path.exists():
            print(f"[violin 4-models tirex] File not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin 4-models tirex] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)

    if not all_data:
        print("[violin 4-models tirex] No data found, skipping.")
        return

    plot_df = pd.concat(all_data, ignore_index=True)
    out_dir = PROJECT_ROOT / "output" / "08_baseline_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col, xlabel, xrot, suffix in [
        ("year", "Year", 0, "per_year"),
        ("month", "Month", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            inner="quartile", cut=0, ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: CV vs EKF vs TiRex vs LSTM ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            out_dir / f"{EVAL_SPLIT}_violin_cv_ekf_tirex_lstm_{suffix}.png",
            xrot=xrot,
        )

    all_data = []
    for path, label in sources:
        if not path.exists():
            print(f"[violin 4-models] File not found: {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if "month" not in df.columns or "RMSE" not in df.columns:
            print(f"[violin 4-models] '{label}' CSV missing 'month' or 'RMSE' column, skipping.")
            continue
        df = df.dropna(subset=["month", "RMSE"])[["month", "RMSE"]].copy()
        df["year"] = df["month"].apply(lambda x: str(x).split("-")[0])
        df["model"] = label
        all_data.append(df)

    if not all_data:
        print("[violin 4-models] No data found, skipping.")
        return

    plot_df = pd.concat(all_data, ignore_index=True)
    out_dir = PROJECT_ROOT / "output" / "08_baseline_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col, xlabel, xrot, suffix in [
        ("year", "Year", 0, "per_year"),
        ("month", "Month", 45, "per_month"),
    ]:
        plot_df_sorted = plot_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(
            data=plot_df_sorted, x=x_col, y="RMSE", hue="model",
            inner="quartile", cut=0, ax=ax, alpha=0.7,
        )
        _save_violin(
            fig, ax,
            f"RMSE: CV vs EKF vs Chronos vs LSTM ({EVAL_SPLIT} split)",
            xlabel, "RMSE [m]",
            out_dir / f"{EVAL_SPLIT}_violin_cv_ekf_chronos_lstm_{suffix}.png",
            xrot=xrot,
        )


if __name__ == "__main__":
    print(f"Using configured run directory: {selected_run_dir}")
    print(f"Reading diagnostics from: {diagnostics_dir}")
    print(f"Saving plots to: {plots_dir}\n")

    plot_ade_horizon()
    plot_monthly_metrics()
    plot_context_metrics()
    plot_uncertainty_metrics()
    plot_channel_importance()
    plot_timestep_importance()
    plot_tuning_results()
    plot_violin_cv_vs_kalman()
    plot_violin_ctrv_vs_ekf()
    plot_violin_chronos_vs_tirex()
    plot_violin_lstm_vs_lstm_domain()
    plot_violin_lstm_vs_nohpo_lstm_domain()
    plot_violin_cv_ekf_chronos_lstm()
    plot_violin_cv_ekf_tirex_lstm()

    print("\nDone.")
