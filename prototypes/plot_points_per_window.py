# AIS Points-per-Window Boxplot
# One subplot per dataset, x = window size, y = point count distribution

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
WINDOWS = [1, 5, 10, 20]  # minutes


def collect_point_counts(df, window_minutes):
    window_s = window_minutes * 60
    counts = []
    for vessel_id, group in df.groupby('vessel_id'):
        group = group.sort_values('t_utc')
        times = group['t_utc'].values.astype('int64') / 1e9
        if len(times) < 2:
            continue
        for i in range(len(times)):
            n = int(np.sum((times >= times[i]) & (times < times[i] + window_s)))
            counts.append(n)
    return counts


def plot_boxplots(datasets: dict, save_path="output/figures/points_per_window.pdf"):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(TEXTWIDTH * n / 2, 3.6), sharey=False)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, (label, df), color in zip(axes, datasets.items(), colors):
        data_per_window = []
        for w in WINDOWS:
            counts = collect_point_counts(df, w)
            data_per_window.append(counts)

        bp = ax.boxplot(
            data_per_window,
            positions=range(len(WINDOWS)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )

        ax.set_xticks(range(len(WINDOWS)))
        ax.set_xticklabels([f'{w}\\,min' for w in WINDOWS])
        ax.set_xlabel('Window size')
        ax.set_ylabel('Points per window')
        ax.set_title(label)

        # Annotate median values on the median line
        for i, d in enumerate(data_per_window):
            med = np.median(d)
            ax.text(i, med, f'{med:.0f}',
                    ha='center', va='bottom', fontsize=7, color='black')

    fig.suptitle(r'AIS Points per Time Window per Vessel', fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    dataset_files = {
        'Kiel 2021-07-01':        'output/02_cleaned/processed_ais_kiel_20210701.csv',
        'Bremerhaven 2018-04-04': 'output/02_cleaned/processed_ais_bremerhaven_20180404.csv',
        'Mississippi 2024-01-01': 'output/02_cleaned/processed_ais_marinecadastre_2024_01.csv',
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
        plot_boxplots(datasets)
        print("Done!")
