# AIS Reporting Interval Analysis - Plot 1 v3
# Class A + B spec lines, NOAA 60s resample mark, no annotations/arrows

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
})

TEXTWIDTH = 5.5
Path("output/figures").mkdir(parents=True, exist_ok=True)

SPEC_A_SOG = [0,  3,  14, 23, 50]
SPEC_A_DT  = [10, 10,  6,  2,  2]

SPEC_B_SOG = [0,   2,  14, 23, 50]
SPEC_B_DT  = [180, 30, 15,  5,  5]

NOAA_DATASETS = {'Mississippi 2024-01-01'}


def compute_intervals(df):
    df = df.sort_values(['vessel_id', 't_utc'])
    df['dt'] = df.groupby('vessel_id')['t_utc'].diff().dt.total_seconds()
    df = df.dropna(subset=['dt', 'sog'])
    df = df[(df['dt'] > 0) & (df['dt'] < 600)]
    return df


def plot_interval_vs_sog(datasets: dict, save_path="output/figures/interval_vs_sog.pdf"):
    bin_edges   = [0, 3, 14, 23, 50]
    bin_centres = [(a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    labels      = [f"{a}--{b}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    colors      = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.8))

    for (label, df), color in zip(datasets.items(), colors):
        df2 = compute_intervals(df.copy())
        df2['sog_bin'] = pd.cut(df2['sog'], bins=bin_edges, labels=labels, right=False)
        stats = df2.groupby('sog_bin', observed=True)['dt'].median().reindex(labels)
        ax.plot(bin_centres, stats.values, marker='o', label=label,
                color=color, linewidth=1.5, markersize=5, zorder=3)

    # ITU-R Class A spec
    ax.step(SPEC_A_SOG, SPEC_A_DT, where='post',
            color='black', linestyle='--', linewidth=1.2,
            label='ITU-R M.1371 Class A', zorder=2)

    # ITU-R Class B spec
    ax.step(SPEC_B_SOG, SPEC_B_DT, where='post',
            color='gray', linestyle=':', linewidth=1.2,
            label='ITU-R M.1371 Class B', zorder=2)

    # NOAA 1-minute resample line
    ax.axhline(60, color='orange', linestyle='-.', linewidth=1.2,
               label='NOAA 1-min resample', zorder=2)

    ax.set_xlabel("Speed over Ground (kn)")
    ax.set_ylabel("Median reporting interval (s)")
    ax.set_title("AIS Reporting Interval vs.\ Speed (ITU-R M.1371-5)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 200)
    ax.legend(frameon=False, loc='upper right')

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

    if datasets:
        plot_interval_vs_sog(datasets)
        print(r"Include with: \includegraphics{output/figures/interval_vs_sog.pdf}")
