# AIS Trajectory Visualizer

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from pathlib import Path

#this script can be used to manually inspect individual tracks
#created with help of Claude Sonnet for interactive plotting

# Globals also togglable in the GUI
SHOW_CV = True
SHOW_CTRV = True
SHOW_CTRV_ARC = True
SHOW_HYBRID = True
SHOW_KALMAN = True
SHOW_CTRV_EKF = True

def load_data(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif Path(source).suffix == '.parquet':
        df = pd.read_parquet(source)
        df['t_utc'] = pd.to_datetime(df['t_utc'])
    else:
        df = pd.read_csv(source)
        df['t_utc'] = pd.to_datetime(df['t_utc'])

    return df


def load_model_predictions(source_file, model_output_dir='output/08_baseline_results/model_output'):
    model_specs = {
        'constant_velocity': {'label': 'CV',     'color': '#ff7f0e', 'marker': 'x', 'alpha': 0.9},
        'ctrv':              {'label': 'CTRV',   'color': '#d62728', 'marker': '^', 'alpha': 0.9},
        'ctrv_arc':          {'label': 'CTRV Arc', 'color': '#e377c2', 'marker': 'v', 'alpha': 0.85},
        'hybrid_cv_ctrv':    {'label': 'Hybrid', 'color': '#9467bd', 'marker': 's', 'alpha': 0.45},#overlaps, transparent
        'kalman':            {'label': 'Kalman', 'color': '#17becf', 'marker': 'D', 'alpha': 0.45}, #overlaps, transparent
        'ctrv_ekf':          {'label': 'CTRV EKF', 'color': '#8c564b', 'marker': 'P', 'alpha': 0.55},
    }

    source_stem = Path(source_file).stem
    output_dir = Path(model_output_dir)
    predictions = {}

    for model_key in model_specs:
        pred_path = output_dir / f"{source_stem}__{model_key}.csv"
        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        if 'track_id' in pred_df.columns:
            pred_df['track_id'] = pred_df['track_id'].astype('int64')
        predictions[model_key] = pred_df

    return predictions, model_specs


def create_interactive_plot(df, model_predictions=None, model_specs=None):
    df = df.sort_values(['track_id', 't_utc']).reset_index(drop=True)

    track_ids = sorted(df['track_id'].unique())
    n_tracks = len(track_ids)

    model_predictions = model_predictions or {}
    model_specs = model_specs or {}

    context_style = {'label': 'Context',      'color': '#1f77b4', 'marker': 'o'}
    control_style = {'label': 'Ground Truth', 'color': '#2ca02c', 'marker': '*'}

    # visibility state, keyed by model_key, initialised from global switches
    visibility = {
        'constant_velocity': SHOW_CV,
        'ctrv':              SHOW_CTRV,
        'ctrv_arc':          SHOW_CTRV_ARC,
        'hybrid_cv_ctrv':    SHOW_HYBRID,
        'kalman':            SHOW_KALMAN,
        'ctrv_ekf':          SHOW_CTRV_EKF,
    }

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.22, left=0.1)

    x_min, x_max = df['x'].min() - 100, df['x'].max() + 100
    y_min, y_max = df['y'].min() - 100, df['y'].max() + 100

    def update_track(val):
        tid_idx = int(slider.val)
        current_tid = track_ids[tid_idx]
        track_data = df[df['track_id'] == current_tid]

        ax.clear()

        ctx = track_data[track_data['role'] == 'context']
        control = track_data[track_data['role'] == 'prediction']

        legend_added = False

        if not ctx.empty:
            ax.plot(ctx['x'], ctx['y'], color=context_style['color'], alpha=0.8, linewidth=2, zorder=1,
                    label=context_style['label'])
            ax.scatter(ctx['x'], ctx['y'], color=context_style['color'], s=40, marker=context_style['marker'],
                       alpha=0.9, edgecolors='darkblue', linewidth=0.5, zorder=2)
            legend_added = True

        if not control.empty:
            ax.plot(control['x'], control['y'], color=control_style['color'], alpha=0.9, linewidth=2.2, zorder=3,
                    label=control_style['label'])
            ax.scatter(control['x'], control['y'], color=control_style['color'], s=90, marker=control_style['marker'],
                       alpha=1.0, edgecolors='black', linewidth=0.8, zorder=4)
            legend_added = True

        x_series = [ctx['x']] if not ctx.empty else []
        y_series = [ctx['y']] if not ctx.empty else []
        if not control.empty:
            x_series.append(control['x'])
            y_series.append(control['y'])

        for model_key, pred_df in model_predictions.items():
            if not visibility.get(model_key, True):
                continue
            style = model_specs.get(model_key, {'label': model_key, 'color': '#9467bd', 'marker': 'x', 'alpha': 0.9})
            alpha = style.get('alpha', 0.9)
            model_track = pred_df[pred_df['track_id'] == current_tid].sort_values('pred_step')

            if model_track.empty:
                continue

            ax.plot(model_track['x_pred'], model_track['y_pred'], color=style['color'], alpha=alpha,
                    linewidth=2, zorder=5, label=style['label'])
            ax.scatter(model_track['x_pred'], model_track['y_pred'], color=style['color'], s=65,
                       marker=style['marker'], alpha=alpha, linewidth=0.8, zorder=6)
            x_series.append(model_track['x_pred'])
            y_series.append(model_track['y_pred'])
            legend_added = True

        if legend_added:
            ax.legend(loc='best')

        ax.set_aspect('equal')
        ax.set_xlabel('X (m, UTM)', fontsize=12)
        ax.set_ylabel('Y (m, UTM)', fontsize=12)
        ax.grid(True, alpha=0.3)

        title = (f'Track ID: {current_tid} | '
                 f'Vessel: {track_data["vessel_id"].iloc[0] if not track_data.empty else "N/A"} | '
                 f'{len(ctx)} context + {len(control)} ground-truth points | '
                 f'Time: {track_data["t_utc"].min().strftime("%H:%M")} - {track_data["t_utc"].max().strftime("%H:%M")}')
        ax.set_title(title, fontsize=13, pad=15)

        if x_series:
            all_x = pd.concat(x_series, ignore_index=True)
            all_y = pd.concat(y_series, ignore_index=True)
            x_pad = (all_x.max() - all_x.min()) * 0.15 or 100
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

    # Model toggle buttons
    toggle_keys   = ['constant_velocity', 'ctrv', 'ctrv_arc', 'hybrid_cv_ctrv', 'kalman', 'ctrv_ekf']
    toggle_labels = ['CV', 'CTRV', 'CTRV Arc', 'Hybrid', 'Kalman', 'CTRV EKF']
    toggle_active = [visibility[k] for k in toggle_keys]

    ax_check = plt.axes([0.15, 0.01, 0.50, 0.08])
    check = CheckButtons(ax_check, toggle_labels, toggle_active)
    # Colour the check-box rectangles to match each model
    for rect, key in zip(ax_check.patches, toggle_keys):
        rect.set_facecolor(model_specs.get(key, {}).get('color', 'gray'))
        rect.set_alpha(0.6)

    def on_toggle(label):
        key = toggle_keys[toggle_labels.index(label)]
        visibility[key] = not visibility[key]
        update_track(slider.val)

    check.on_clicked(on_toggle)

    # Track ID slider
    ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    slider = Slider(ax_slider, f'Track (0–{n_tracks - 1})', 0, n_tracks - 1,
                    valinit=0, valfmt='%0.0f', valstep=1)

    # Reset button
    ax_reset = plt.axes([0.85, 0.01, 0.12, 0.04])
    reset_btn = Button(ax_reset, 'Reset View', color='lightgray', hovercolor='gray')
    reset_btn.on_clicked(reset_view)

    slider.on_changed(update_track)

    update_track(0)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    # Point to a parquet file in output/07_parquet/ (stem must match the prediction output filenames)
    source_file = project_root / 'output/07_parquet/test_harbour_processed_kiel_AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Kiel-2025-01_01.parquet'
    #source_file = project_root / 'output/07_parquet/test_harbour_processed_kiel_AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Kiel-2025-07_01.parquet'
    #source_file = project_root / 'output/07_parquet/test_river_processed_bremerhaven_AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Bremerhaven-2025-02_16.parquet'

    df = load_data(source_file)
    model_predictions, model_specs = load_model_predictions(source_file, model_output_dir=project_root / 'output/08_baseline_results/model_output')
    create_interactive_plot(df, model_predictions=model_predictions, model_specs=model_specs)
