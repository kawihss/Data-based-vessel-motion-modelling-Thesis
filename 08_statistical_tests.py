from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

PROJECT_ROOT = Path(__file__).resolve().parent

RUN_DIRS = [
    PROJECT_ROOT / "output/08_baseline_results/runs/baseline_full_full_100",
    #PROJECT_ROOT / "output/08_baseline_results/runs/foundation_full_full_100",
]
OUTPUT_DIR = PROJECT_ROOT / "output/08_baseline_results/statistical_tests/full_full"
SPLIT = "test"
ALPHA = 0.05

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
    print("\n".join(f"  {r.mean_rank:.3f}  {r.model_key}" for _, r in ranks_df.iterrows()))

    # nemenyi post-hoc
    if significant:
        nemenyi_p = sp.posthoc_nemenyi_friedman(matrix)
        nemenyi_p.to_csv(OUTPUT_DIR / "nemenyi_pvalues.csv", float_format="%.16e")
        (nemenyi_p < ALPHA).astype(int).to_csv(OUTPUT_DIR / "nemenyi_significant.csv")
    # plot 1: mean ranks bar chart 
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(ranks_df) + 1.2))
    bars = ax.barh(ranks_df["model_key"][::-1], ranks_df["mean_rank"][::-1], color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel("Mean rank (lower = better)")
    ax.set_title(f"Friedman mean ranks  —  χ²={chi2:.2f}, p={p_value:.2e}")
    ax.set_xlim(1, len(matrix.columns) + 0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_ranks.png", dpi=150)
    plt.close(fig)

    
    #plot 2: critical difference diagram
    if significant:
        fig, ax = plt.subplots(figsize=(8, 2 + 0.35 * len(mean_ranks)))
        sp.critical_difference_diagram(mean_ranks, nemenyi_p, ax=ax, alpha=ALPHA)
        ax.set_title(f"Critical Difference diagram  (Nemenyi, α={ALPHA},  n={len(matrix)} blocks)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "cd_diagram.png", dpi=150)
        plt.close(fig)
