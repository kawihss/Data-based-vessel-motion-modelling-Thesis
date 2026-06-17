"""
generated script.
Naive check whether prediction errors follow a Gaussian distribution.

Loads all model-output CSVs from a run directory, computes signed errors
(dx = x_pred - x_gt, dy = y_pred - y_gt), and shows:
  - histogram + fitted normal curve
  - QQ-plot
  - Shapiro-Wilk p-value
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

RUN_DIR = "output/08_baseline_results/runs/baseline_full_full_100/model_output"
# filter to one model to keep it focused; set to None to use all models
MODEL_FILTER = "constant_velocity"   # e.g. "ctrv", "kalman", "constant_velocity", or None
MAX_ROWS = 200_000      # cap for Shapiro-Wilk (needs <= 5000; subsampled below)

# ── load ─────────────────────────────────────────────────────────────────────
pattern = os.path.join(RUN_DIR, "*.csv")
if MODEL_FILTER:
    pattern_list = glob.glob(os.path.join(RUN_DIR, f"*__{MODEL_FILTER}.csv"))
else:
    pattern_list = glob.glob(os.path.join(RUN_DIR, "*.csv"))

print(f"Loading {len(pattern_list)} files ...")
df = pd.concat(
    [pd.read_csv(f, usecols=["x_pred", "y_pred", "x_gt", "y_gt", "model_key"])
     for f in pattern_list],
    ignore_index=True,
)
print(f"  {len(df):,} rows total")

dx = (df["x_pred"] - df["x_gt"]).dropna().values
dy = (df["y_pred"] - df["y_gt"]).dropna().values

model_label = MODEL_FILTER or "all models"

# ── Shapiro-Wilk (subsample if needed, test only works up to n=5000) ─────────
def shapiro_info(data, name):
    sample = data if len(data) <= 5000 else np.random.default_rng(0).choice(data, 5000, replace=False)
    stat, p = stats.shapiro(sample)
    print(f"  Shapiro-Wilk {name}: stat={stat:.4f}, p={p:.4e}  "
          f"({'likely normal' if p > 0.05 else 'NOT normal'})")

print(f"\nShapiro-Wilk test (subsample n=5000 if needed):")
shapiro_info(dx, "dx")
shapiro_info(dy, "dy")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Error distribution — {model_label}", fontsize=13)

for col_idx, (errors, label) in enumerate([(dx, "dx (x error)"), (dy, "dy (y error)")]):
    mu, sigma = errors.mean(), errors.std()

    # histogram + fitted Gaussian
    ax = axes[0, col_idx]
    ax.hist(errors, bins=80, density=True, alpha=0.6, color="steelblue", label="observed")
    x_range = np.linspace(errors.min(), errors.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r-", lw=2,
            label=f"N(μ={mu:.1f}, σ={sigma:.1f})")
    ax.set_title(f"Histogram: {label}")
    ax.set_xlabel("error [m]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # QQ-plot
    ax = axes[1, col_idx]
    sample = errors if len(errors) <= 10_000 else np.random.default_rng(1).choice(errors, 10_000, replace=False)
    (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")
    ax.scatter(osm, osr, s=2, alpha=0.3, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2, label=f"R²={r**2:.4f}")
    ax.set_title(f"QQ-plot: {label}")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("sample quantiles")
    ax.legend(fontsize=8)

plt.tight_layout()
out_path = f"plots/error_distribution_{MODEL_FILTER or 'all'}.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()
