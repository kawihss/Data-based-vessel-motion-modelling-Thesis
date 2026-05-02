import numpy as np
import matplotlib.pyplot as plt


def _require_quantile(quantile_predictions, quantile):
    if quantile not in quantile_predictions:
        raise ValueError(f"Required quantile {quantile} missing from prediction set.")
    return np.asarray(quantile_predictions[quantile], dtype=float)

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


def calculate_mean_interval_width(y_quantiles, lower_q=0.1, upper_q=0.9):
    lower = _require_quantile(y_quantiles, lower_q)
    upper = _require_quantile(y_quantiles, upper_q)
    widths = np.sqrt(((upper - lower) ** 2).sum(axis=-1))
    return float(widths.mean())


def calculate_coverage(y_true, y_quantiles, lower_q=0.1, upper_q=0.9):
    lower = _require_quantile(y_quantiles, lower_q)
    upper = _require_quantile(y_quantiles, upper_q)
    inside = ((y_true >= lower) & (y_true <= upper)).all(axis=-1)
    return float(inside.mean())


def calculate_crps_approximation(y_true, y_quantiles):
    quantiles = np.array(sorted(y_quantiles), dtype=float)
    losses = np.array([
        np.maximum(
            quantile * (y_true - np.asarray(y_quantiles[quantile], dtype=float)),
            (quantile - 1.0) * (y_true - np.asarray(y_quantiles[quantile], dtype=float)),
        ).mean()
        for quantile in quantiles
    ], dtype=float)
    return float(2.0 * np.trapezoid(losses, quantiles))


def calculate_winkler_score(y_true, y_quantiles, lower_q=0.1, upper_q=0.9):
    lower = _require_quantile(y_quantiles, lower_q)
    upper = _require_quantile(y_quantiles, upper_q)
    alpha = 1.0 - (upper_q - lower_q)
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    penalty = ((2.0 / alpha) * (lower - y_true) * below) + ((2.0 / alpha) * (y_true - upper) * above)
    return float((width + penalty).mean())


def evaluate_quantile_forecast(y_true, y_quantiles):
    return {
        "MIW": calculate_mean_interval_width(y_quantiles, lower_q=0.1, upper_q=0.9),
        "Coverage": calculate_coverage(y_true, y_quantiles, lower_q=0.1, upper_q=0.9),
        "IQR": calculate_mean_interval_width(y_quantiles, lower_q=0.25, upper_q=0.75),
        "CRPSApprox": calculate_crps_approximation(y_true, y_quantiles),
        "Winkler80": calculate_winkler_score(y_true, y_quantiles, lower_q=0.1, upper_q=0.9),
    }


def calculate_channel_importance(covariate_values, motion_magnitude):
    target = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in motion_magnitude])
    raw_importance = {}
    for covariate, values in covariate_values.items():
        series = np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in values])
        corr = np.corrcoef(series, target)[0, 1]
        if not np.isfinite(corr):
            raise ValueError(f"Pearson correlation for covariate '{covariate}' is not finite.")
        raw_importance[covariate] = abs(float(corr))

    total = float(sum(raw_importance.values()))
    if total <= 0.0:
        raise ValueError("Channel importance cannot be normalized because the total absolute correlation is zero.")

    return {covariate: value / total for covariate, value in raw_importance.items()}

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