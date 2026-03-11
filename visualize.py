# AIS Trajectory Visualizer - Generated using perplexity with Claude Sonnes 4.6 Thinking

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path


def load_data(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        print(f"Loading data from: {source}")
        df = pd.read_csv(source)
        df['t_utc'] = pd.to_datetime(df['t_utc'])

    print(f"Loaded {len(df):,} records | "
          f"{df['vessel_id'].nunique()} vessels | "
          f"{df['track_id'].nunique()} segments")
    return df


def create_interactive_plot(df):
    df = df.sort_values(['track_id', 't_utc']).reset_index(drop=True)

    track_ids = sorted(df['track_id'].unique())
    n_tracks = len(track_ids)

    # Precompute colors per track
    cmap = plt.cm.get_cmap('tab20')
    track_color = {tid: cmap(i % 20) for i, tid in enumerate(track_ids)}

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.2, left=0.1)

    x_min, x_max = df['x'].min() - 100, df['x'].max() + 100
    y_min, y_max = df['y'].min() - 100, df['y'].max() + 100

    def update_track(val):
        tid_idx = int(slider.val)
        current_tid = track_ids[tid_idx]
        track_data = df[df['track_id'] == current_tid]

        ax.clear()

        ctx = track_data[track_data['role'] == 'context']
        pred = track_data[track_data['role'] == 'prediction']

        color = track_color[current_tid]

        # Plot context: line + scatter
        if not ctx.empty:
            ax.plot(ctx['x'], ctx['y'], color=color, alpha=0.7, linewidth=2, zorder=1)
            ax.scatter(ctx['x'], ctx['y'], color=color, s=40, marker='o',
                    alpha=0.9, edgecolors='darkblue', linewidth=0.5, zorder=2)

        # Plot predictions: stars
        if not pred.empty:
            ax.scatter(pred['x'], pred['y'], color=color, s=100, marker='*',
                    alpha=1.0, edgecolors='black', linewidth=1.0, zorder=3)

        # Labels and formatting
        ax.set_aspect('equal')
        ax.set_xlabel('X (m, UTM)', fontsize=12)
        ax.set_ylabel('Y (m, UTM)', fontsize=12)
        ax.grid(True, alpha=0.3)

        title = (f'Track ID: {current_tid} | '
                f'Vessel: {track_data["vessel_id"].iloc[0] if not track_data.empty else "N/A"} | '
                f'{len(ctx)} context + {len(pred)} prediction points | '
                f'Time: {track_data["t_utc"].min().strftime("%H:%M")} - {track_data["t_utc"].max().strftime("%H:%M")}')
        ax.set_title(title, fontsize=13, pad=15)

        # Auto zoom to ALL POINTS in track (ctx + pred) + margin
        if not track_data.empty:
            all_x = track_data['x']
            all_y = track_data['y']
            x_pad = (all_x.max() - all_x.min()) * 0.15 or 100  # 15% margin
            y_pad = (all_y.max() - all_y.min()) * 0.15 or 100
            ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
            ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.canvas.draw_idle()


    def reset_view(event):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m, UTM)')
        ax.set_ylabel('Y (m, UTM)')
        ax.set_title('Full View Reset')
        fig.canvas.draw_idle()

    # Track ID slider
    ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(ax_slider, f'Track ID (0 to {n_tracks-1})', 0, n_tracks-1,
                    valinit=0, valfmt='%0.0f', valstep=1)

    # Reset button
    ax_reset = plt.axes([0.85, 0.01, 0.12, 0.04])
    reset_btn = Button(ax_reset, 'Reset View', color='lightgray', hovercolor='gray')
    reset_btn.on_clicked(reset_view)

    slider.on_changed(update_track)

    # Initial plot
    update_track(0)

    print(f"Track ID slider: 0 to {n_tracks-1} ({n_tracks} total tracks)")
    print("Each track: context (dots + line) → prediction (stars)")
    plt.show()


def create_track_summary(df, save_path='output/track_summary.png'):
    """Simplified track length summary"""
    lengths = df.groupby('track_id').size()
    ctx_count = df[df['role'] == 'context'].groupby('track_id').size()
    pred_count = df[df['role'] == 'prediction'].groupby('track_id').size()
    
    print(f"\n=== TRACK SUMMARY ===")
    print(f"Avg track length: {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Tracks with 72 points: {(lengths == 72).sum()}")
    print(f"Avg context points: {ctx_count.mean():.1f}")
    print(f"Avg prediction points: {pred_count.mean():.1f}")
    print(f"Tracks with 12 pred points: {(pred_count == 12).sum()}")
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    lengths.hist(bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(72, color='red', linestyle='--', label='Expected: 72')
    ax.set_xlabel('Points per track')
    ax.set_ylabel('Number of tracks')
    ax.set_title('Distribution of Track Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved: {save_path}")
    plt.show()


if __name__ == "__main__":

    # Option 1: Load from CSV
    csv_file = 'output/04_trajectories/train_processed_ais_kiel_20210701.csv'
   #csv_file = 'output/04_trajectories/train_processed_ais_marinecadastre_2024_01.csv'

    if Path(csv_file).exists():
        df = load_data(csv_file)
        create_track_summary(df)
        print("\nOpening Track ID visualizer...")
        create_interactive_plot(df)
    else:
        print(f"ERROR: File not found: {csv_file}")
