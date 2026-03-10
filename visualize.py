# AIS Trajectory Visualizer with Time Slider (ZOOM-PRESERVED VERSION)
# Interactive visualization of vessel positions over time
# Updated for new column format: t_utc, vessel_id, lat, lon

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from pathlib import Path


def load_data(source):
    """
    Load AIS data from CSV or use existing DataFrame

    Args:
        source: Either a DataFrame or path to CSV file

    Returns:
        DataFrame with AIS data
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        print(f"Loading data from: {source}")
        df = pd.read_csv(source)
        df['t_utc'] = pd.to_datetime(df['t_utc'])
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].astype('category')
        if 'nav_status' in df.columns:
            df['nav_status'] = df['nav_status'].astype('category')

    print(f"Loaded {len(df):,} records from {df['vessel_id'].nunique()} vessels")
    print(f"Time range: {df['t_utc'].min()} to {df['t_utc'].max()}")

    return df


def create_interactive_plot(df, time_window_minutes=5):
    """
    Create interactive plot with time slider (preserves zoom)

    Args:
        df: DataFrame with AIS data
        time_window_minutes: Time window to display (in minutes)
    """

    # Sort by timestamp
    df = df.sort_values('t_utc').reset_index(drop=True)

    # Create time bins (every minute)
    start_time = df['t_utc'].min()
    end_time = df['t_utc'].max()
    duration_minutes = int((end_time - start_time).total_seconds() / 60)

    print(f"\nTotal duration: {duration_minutes} minutes")
    print(f"Time window: {time_window_minutes} minutes")

    # Store initial bounds
    lon_min, lon_max = df['lon'].min() - 0.01, df['lon'].max() + 0.01
    lat_min, lat_max = df['lat'].min() - 0.01, df['lat'].max() + 0.01

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    # Track whether to auto-adjust view
    auto_adjust = {'enabled': False}

    def get_data_at_time(center_time, window_minutes):
        """Get data within time window"""
        time_delta = pd.Timedelta(minutes=window_minutes/2)
        mask = (df['t_utc'] >= center_time - time_delta) & \
               (df['t_utc'] <= center_time + time_delta)
        return df[mask]

    # Function to update plot
    def update(val):
        # Get current time from slider (in minutes from start)
        minutes_offset = slider.val
        current_time = start_time + pd.Timedelta(minutes=minutes_offset)

        # Get data for current time window
        data = get_data_at_time(current_time, time_window_minutes)

        # Store current axis limits BEFORE clearing
        if not auto_adjust['enabled']:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        # Clear axis
        ax.clear()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No vessels in this time window',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='red')
        else:
            # Plot by dataset if available
            if 'dataset' in data.columns and data['dataset'].nunique() > 1:
                for dataset in data['dataset'].unique():
                    dataset_data = data[data['dataset'] == dataset]
                    ax.scatter(dataset_data['lon'], dataset_data['lat'],
                             alpha=0.6, s=30, label=dataset, edgecolors='black', linewidth=0.5)
                ax.legend(loc='upper right')
            else:
                # Color by vessel if single dataset
                for vessel_id in data['vessel_id'].unique()[:20]:  # Limit to 20 vessels
                    vessel_data = data[data['vessel_id'] == vessel_id]
                    ax.scatter(vessel_data['lon'], vessel_data['lat'],
                             alpha=0.6, s=30, label=f'{vessel_id}', edgecolors='black', linewidth=0.5)
                if data['vessel_id'].nunique() <= 20:
                    ax.legend(loc='upper right', fontsize=8, ncol=2)

        # Set labels and title
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(f'Vessel Positions | {current_time.strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'{len(data)} positions from {data["vessel_id"].nunique() if len(data) > 0 else 0} vessels',
                    fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)

        # Restore axis limits (preserves zoom) or reset to full view
        if auto_adjust['enabled']:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
        else:
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except:
                # First time, set to full view
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)

        fig.canvas.draw_idle()

    # Function to reset view
    def reset_view(event):
        auto_adjust['enabled'] = True
        update(slider.val)
        auto_adjust['enabled'] = False
        print("View reset to full extent")

    # Create slider
    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Time (minutes)',
        valmin=0,
        valmax=duration_minutes,
        valinit=0,
        valstep=1
    )

    # Create reset button
    ax_button = plt.axes([0.8, 0.01, 0.1, 0.04])
    button = Button(ax_button, 'Reset View', color='lightgray', hovercolor='gray')
    button.on_clicked(reset_view)

    # Connect slider to update function
    slider.on_changed(update)

    # Initial plot
    auto_adjust['enabled'] = True
    update(0)
    auto_adjust['enabled'] = False

    print("\nControls:")
    print("  • Drag slider to move through time")
    print("  • Scroll to zoom in/out")
    print("  • Click and drag to pan")
    print("  • Click 'Reset View' button to see full extent")

    plt.show()


def create_static_overview(df, save_path='output/ais_overview.png'):
    """
    Create static overview plot of all trajectories

    Args:
        df: DataFrame with AIS data
        save_path: Path to save the plot
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All trajectories
    ax1 = axes[0]
    if 'dataset' in df.columns and df['dataset'].nunique() > 1:
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            ax1.scatter(dataset_data['lon'], dataset_data['lat'],
                       alpha=0.3, s=5, label=dataset)
        ax1.legend()
    else:
        ax1.scatter(df['lon'], df['lat'], alpha=0.3, s=5, c='blue')

    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title(f'All Vessel Positions ({len(df):,} points)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Vessel count over time
    ax2 = axes[1]
    df_time = df.set_index('t_utc')
    vessel_counts = df_time.groupby(pd.Grouper(freq='10Min'))['vessel_id'].nunique()
    ax2.plot(vessel_counts.index, vessel_counts.values, linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Active Vessels')
    ax2.set_title('Vessel Activity Over Time (10-min bins)')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nStatic overview saved to: {save_path}")
    plt.close()


# Usage
if __name__ == "__main__":

    # Option 1: Load from CSV
    csv_file = 'output/02_cleaned/processed_ais_combined.csv'
    #csv_file = 'output/02_cleaned/processed_ais_kiel_20210701.csv'
    #csv_file = 'output/02_cleaned/processed_ais_marinecadastre_2024_01.csv'

    if Path(csv_file).exists():
        df = load_data(csv_file)

        # Create static overview
        create_static_overview(df)

        # Create interactive plot
        print("\nOpening interactive visualization...")
        create_interactive_plot(df, time_window_minutes=5)
    else:
        print(f"ERROR: File not found: {csv_file}")
        print("\nRun load_ais_multi_dataset_formatted.py first to generate the data!")
