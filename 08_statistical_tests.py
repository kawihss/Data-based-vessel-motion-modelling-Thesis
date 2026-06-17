from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

PROJECT_ROOT = Path(__file__).resolve().parent

RUN_DIRS = [ # EXPERIMENT 1 full_full_100
    PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_full_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_full_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/lstm_full_full_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_full_full_100",
    #PROJECT_ROOT / "output/08_baseline_results/runs/noHPO_OHE_lstm_full_full_100", # 2 x same name breaks it
    #add comp for n_n etc untereinander?..
]


# RUN_DIRS = [ # EXPERIMENT 2 harbour
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_harbour_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_harbour_harbour_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_harbour_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_harbour_harbour_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_full_harbour_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_harbour_harbour_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_full_harbour_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_harbour_harbour_100",
# ]

RUN_DIRS = [ # EXPERIMENT 2 channel
    PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_channel_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/baseline_channel_channel_100",

    PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_channel_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/foundation_channel_channel_100",

    PROJECT_ROOT / "output/08_baseline_results/runs/lstm_full_channel_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/lstm_channel_channel_100",

    PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_full_channel_100",
    PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_channel_channel_100",
]

# RUN_DIRS = [ # EXPERIMENT 2 lock
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_lock_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_lock_lock_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_lock_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_lock_lock_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_full_lock_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_lock_lock_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_full_lock_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_lock_lock_100",
# ]

# RUN_DIRS = [ # EXPERIMENT 2 river
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_river_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/baseline_river_river_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_river_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/foundation_river_river_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_full_river_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/lstm_river_river_100",
#
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_full_river_100",
#     PROJECT_ROOT / "output/08_baseline_results/runs/OHE_lstm_river_river_100",
# ]

OUTPUT_DIR = PROJECT_ROOT / "output/08_baseline_results/statistical_tests/full_full"
SPLIT = "test"
ALPHA = 0.05

LABELS = {
    "minimal_lstm": "LSTM",
    "minimal_lstm_domain": "LSTM + Domain",
    "ctrv_ekf": "EKF",
    "chronos2_zero_shot": "Chronos2",
    "constant_velocity": "CV",
    "ctrv_arc": "CTRV (Arc)",
    "ctrv": "CTRV",
    "kalman": "KF",
    "hybrid_cv_ctrv": "Hybrid",
    "tirex_lstm": "TiREX",
}

MODEL_COLORS = {
    "LSTM":          "#1f77b4",
    "LSTM + Domain": "#299d82",
    "EKF":           "#ff7f0e",
    "Chronos2":      "#2ca02c",
    "CV":            "#d62728",
    "CTRV (Arc)":    "#9467bd",
    "CTRV":          "#8c564b",
    "KF":            "#e377c2",
    "Hybrid":        "#7f7f7f",
    "TiREX":         "#bcbd22",
}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load per-file RMSE 
    prefix = f"{SPLIT}_metrics_per_file_"
    frames = {}
    for run_dir in RUN_DIRS:
        for csv_path in sorted((run_dir / "diagnostics").glob(f"{prefix}*.csv")):
            model_key = csv_path.stem[len(prefix):]
            df = pd.read_csv(csv_path, usecols=["file", "RMSE"])
            frames[model_key] = df.set_index("file")["RMSE"]

    # block × model matrix (inner join: only files common to all models)
    matrix = pd.concat(frames, axis=1, join="inner")

    # friedman test
    chi2, p_value = friedmanchisquare(*[matrix[col].values for col in matrix.columns])
    significant = p_value < ALPHA

    pd.DataFrame([{"chi2": chi2, "df": len(matrix.columns) - 1, "p_value": p_value,
                   "alpha": ALPHA, "n_blocks": len(matrix), "significant": significant}]
                 ).to_csv(OUTPUT_DIR / "friedman_result.csv", index=False)
    #chi2 -> Friedman chi-squared statistic
    #df -> degrees of freedom (number of models - 1)
    #p_value -> p-value of the test
    #alpha -> significance level used to determine if the result is significant
    #n_blocks -> number of blocks (files) used in the test
    #significant -> boolean indicating whether the result is statistically significant (p < alpha)
    
    # mean ranks (rank 1 = best = lowest RMSE per block)
    mean_ranks = matrix.rank(axis=1, method="average").mean(axis=0).sort_values()
    ranks_df = mean_ranks.reset_index()
    ranks_df.columns = ["model_key", "mean_rank"]
    ranks_df.to_csv(OUTPUT_DIR / "average_ranks.csv", index=False)
    print("\n".join(f"  {r.mean_rank:.3f}  {LABELS.get(r.model_key, r.model_key)}" for _, r in ranks_df.iterrows()))

    # nemenyi post-hoc
    if significant:
        nemenyi_p = sp.posthoc_nemenyi_friedman(matrix)
        nemenyi_p.to_csv(OUTPUT_DIR / "nemenyi_pvalues.csv", float_format="%.16e")
        (nemenyi_p < ALPHA).astype(int).to_csv(OUTPUT_DIR / "nemenyi_significant.csv")
    # critical difference diagram
    if significant:
        mean_ranks_labeled = mean_ranks.rename(index=LABELS)
        nemenyi_p_labeled = nemenyi_p.rename(index=LABELS, columns=LABELS)
        fig, ax = plt.subplots(figsize=(8, 2 + 0.35 * len(mean_ranks)))
        sp.critical_difference_diagram(mean_ranks_labeled, nemenyi_p_labeled, ax=ax, alpha=ALPHA,
                                       color_palette=MODEL_COLORS)
        ax.set_title(f"Critical Difference diagram  (Nemenyi, α={ALPHA})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "cd_diagram.png", dpi=150)
        plt.close(fig)
        print(f"Output saved to: {OUTPUT_DIR}")
