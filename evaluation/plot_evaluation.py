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
# and generates all plots for model comparison:
# - Evaluation quality plots (horizon, monthly, context, uncertainty)
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
    plot_violin_cv_vs_kalman()
    plot_violin_ctrv_vs_ekf()
    plot_violin_chronos_vs_tirex()
    plot_violin_lstm_vs_lstm_domain()
    plot_violin_lstm_vs_nohpo_lstm_domain()
    plot_violin_cv_ekf_chronos_lstm()
    plot_violin_cv_ekf_tirex_lstm()

    print("\nDone.")
