import numpy as np
import matplotlib.pyplot as plt

def calculate_ade(y_true, y_pred):
    # y_true, y_pred: (n_tracks, n_steps, 2), reconstructed x/y positions
    # Average Displacement Error, mean Euclidean error across all steps and tracks
    errors = np.sqrt(((y_true - y_pred) ** 2).sum(axis=-1))
    return float(errors.mean())

def calculate_fde(y_true, y_pred):
    # Final Displacement Error, Euclidean error at the last predicted step
    errors = np.sqrt(((y_true[:, -1, :] - y_pred[:, -1, :]) ** 2).sum(axis=-1))
    return float(errors.mean())

def calculate_rmse(y_true, y_pred):
    # RMSE over all positions (x and y treated jointly)
    mse = ((y_true - y_pred) ** 2).sum(axis=-1).mean()
    return float(np.sqrt(mse))

def calculate_ade_per_step(y_true, y_pred):
    # ADE(t), mean Euclidean error at each prediction step across all tracks
    # returns array of shape (n_steps,)
    errors = np.sqrt(((y_true - y_pred) ** 2).sum(axis=-1))
    return errors.mean(axis=0)

def plot_ade_over_horizon(ade_per_step, step_duration_s=30, label=None, ax=None, save_path=None):
    # Plots ADE(t), error development over the prediction horizon
    # step_duration_s, seconds between prediction steps (default 30s from preprocessing)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    n_steps = len(ade_per_step)
    time_axis = np.arange(1, n_steps + 1) * step_duration_s / 60

    ax.plot(time_axis, ade_per_step, marker='o', markersize=4, label=label)
    ax.set_xlabel('Prediction horizon (min)')
    ax.set_ylabel('ADE (m)')
    ax.set_title('Error development over prediction horizon')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax

def evaluate_trajectory(y_true, y_pred):
    return {
        "ADE": calculate_ade(y_true, y_pred),
        "FDE": calculate_fde(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "ADE_per_step": calculate_ade_per_step(y_true, y_pred),
    }