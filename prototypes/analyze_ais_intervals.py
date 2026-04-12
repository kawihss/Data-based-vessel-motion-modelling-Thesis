# AIS Reporting Interval Analysis
# Visualizes observed update frequency vs ITU-R M.1371-5 spec

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

mpl.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

TEXTWIDTH  = 5.5   
TEXTHEIGHT = 3.5

Path("output/figures").mkdir(parents=True, exist_ok=True)

SPEC_BANDS_A = [
    (0,   3,   180,  "Anchored $\\leq$3 kn"),   # 3 min
    (0,   14,  10,   "0--14 kn"),
    (14,  23,  6,    "14--23 kn"),
    (23,  999, 2,    "$>$23 kn"),
]
SPEC_BANDS_B = [
    (0,   2,   180,  "Class B $\\leq$2 kn"),
    (2,   14,  30,   "Class B 2--14 kn"),
    (14,  23,  15,   "Class B 14--23 kn"),
    (23,  999, 5,    "Class B $>$23 kn"),
]


def compute_intervals(df):
    df = df.sort_values(['vessel_id', 't_utc'])
    df['dt'] = (df.groupby('vessel_id')['t_utc']
                  .diff()
                  .dt.total_seconds())
    # Drop first message per vessel (NaN dt) and implausible values
    df = df.dropna(subset=['dt'])
    df = df[(df['dt'] > 0) & (df['dt'] < 600)]   # cap at 10 min
    return df


def sog_bin(df, bin_edges=None):
    if bin_edges is None:
        bin_edges = [0, 1, 3, 7, 14, 23, 35, 50]
    labels = [f"{a}--{b}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    df['sog_bin'] = pd.cut(df['sog'], bins=bin_edges, labels=labels, right=False)
    return df, labels


# PLOT 1 — Median reporting interval vs SOG  (one line per dataset)
def plot_interval_vs_sog(datasets: dict, save_path="output/figures/interval_vs_sog.pdf"):

    bin_edges = [0, 3, 14, 23, 50]
    bin_centres = [1.5, 8.5, 18.5, 36.5]  # midpoints of each spec band

    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTHEIGHT))

    for (label, df), color in zip(datasets.items(), colors):
        df2 = compute_intervals(df.copy())
        df2, labels = sog_bin(df2, bin_edges)
        stats = (df2.groupby('sog_bin', observed=True)['dt']
                    .median()
                    .reindex(labels))
        ax.plot(bin_centres, stats.values, marker='o', label=label,
                color=color, linewidth=1.5, markersize=4)

    # ITU-R spec reference lines (Class A)
    spec_sog    = [0, 3, 14, 23, 50]
    spec_dt_a   = [10, 10, 6, 2, 2]
    ax.step(spec_sog, spec_dt_a, where='post', color='black',
            linestyle='--', linewidth=1, label='ITU-R M.1371 Class A')

    ax.set_xlabel("Speed over Ground (knots)")
    ax.set_ylabel("Median reporting interval (s)")
    ax.set_title("AIS Reporting Interval vs.\ Speed")
    ax.legend(frameon=False)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 200)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# PLOT 2 — Distribution of reporting intervals (histogram) per dataset
def plot_interval_histogram(datasets: dict, save_path="output/figures/interval_hist.pdf"):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(TEXTWIDTH * n / 2, TEXTHEIGHT),
                              sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, datasets.items()):
        df2 = compute_intervals(df.copy())
        ax.hist(df2['dt'], bins=60, range=(0, 120),
                color='steelblue', edgecolor='none', alpha=0.85)

        # ITU-R reference verticals
        for sog_lo, sog_hi, interval, desc in SPEC_BANDS_A:
            if interval <= 120:
                ax.axvline(interval, color='tomato', linestyle='--',
                           linewidth=0.8, alpha=0.7)

        ax.set_xlabel("Reporting interval (s)")
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.set_title(label.replace('_', r'\_'))

    fig.suptitle("Distribution of AIS Reporting Intervals", y=1.02)
    # shared legend for ITU lines
    patch = mpatches.Patch(color='tomato', label='ITU-R M.1371 Class A thresholds',
                           linestyle='--', fill=False)
    fig.legend(handles=[patch], loc='lower center', frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# PLOT 3 — Median interval per vessel (violin/box per SOG bin)
def plot_interval_boxplot(datasets: dict, save_path="output/figures/interval_boxplot.pdf"):
    fig, axes = plt.subplots(1, len(datasets),
                              figsize=(TEXTWIDTH, TEXTHEIGHT + 0.5),
                              sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bin_edges = [0, 3, 14, 23, 50]

    for ax, (label, df) in zip(axes, datasets.items()):
        df2 = compute_intervals(df.copy())
        df2, labels = sog_bin(df2, bin_edges)

        groups = [df2[df2['sog_bin'] == lbl]['dt'].dropna().values
                  for lbl in labels]
        groups = [g for g in groups if len(g) > 10]
        lbl_used = [lbl for lbl, g in zip(labels,
                    [df2[df2['sog_bin'] == l]['dt'].dropna().values
                     for l in labels]) if len(g) > 10]

        bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                        medianprops=dict(color='tomato', linewidth=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)

        ax.set_xticklabels(lbl_used, rotation=30, ha='right')
        ax.set_xlabel("SOG bin (kn)")
        ax.set_title(label.replace('_', r'\_'))

    axes[0].set_ylabel("Reporting interval (s)")
    fig.suptitle("Reporting Interval Distribution by Speed Bin", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":

    dataset_files = {
        'Kiel 2021-07-01':        'output/processed_ais_kiel_20210701.csv',
        'Bremerhaven 2018-04-04': 'output/processed_ais_bremerhaven_20180404.csv',
        'Mississippi 2024-01-01': 'output/processed_ais_marinecadastre_2024_01.csv',
    }

    datasets = {}
    for label, path in dataset_files.items():
        if Path(path).exists():
            df = pd.read_csv(path, parse_dates=['t_utc'])
            datasets[label] = df
            print(f"Loaded {label}: {len(df):,} records")
        else:
            print(f"WARNING: {path} not found, skipping")

    if not datasets:
        print("No datasets found. Run load_ais_integrated_fast.py first.")
    else:
        #plot_interval_vs_sog(datasets)# separate script
        plot_interval_histogram(datasets)
        plot_interval_boxplot(datasets)
        print("\nAll figures saved to output/figures/")
        print("Include in LaTeX with:")
        #print(r"  \includegraphics{output/figures/interval_vs_sog.pdf}")
        print(r"  \includegraphics{output/figures/interval_hist.pdf}")
        print(r"  \includegraphics{output/figures/interval_boxplot.pdf}")
