import numpy as np
from .base_model import BaselineModel


class ConstantVelocityModel(BaselineModel):
    def __init__(self, velocity_fraction=0.4):
        super().__init__("ConstantVelocity")
        # Fraction of recent context steps to average for velocity estimate (e.g. 0.05 for last 5%)
        self.velocity_fraction = velocity_fraction

    def predict(self, context_df, n_pred_steps):
        # Compute dx, dy from raw x, y positions sorted by time
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values

        dx = np.diff(x)
        dy = np.diff(y)

        # Use last velocity_fraction of steps (at least 1)
        n = max(1, int(np.ceil(len(dx) * self.velocity_fraction)))
        mean_dx = dx[-n:].mean() if n > 0 else 0.0
        mean_dy = dy[-n:].mean() if n > 0 else 0.0

        # Repeat constant displacement for every future step
        # Returns shape (n_pred_steps, 2) in meters
        return np.tile([mean_dx, mean_dy], (n_pred_steps, 1))


class ConstantTurnRateVelocityModel(BaselineModel):
    def __init__(self, velocity_fraction=0.4):
        super().__init__("CTRV")
        # Fraction of recent context steps used to estimate displacement and turn rate
        self.velocity_fraction = velocity_fraction

    def predict(self, context_df, n_pred_steps):
        if n_pred_steps <= 0:
            return np.empty((0, 2), dtype=float)

        # Estimate state from raw context sorted by time
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values

        dx = np.diff(x)
        dy = np.diff(y)
        n = max(1, int(np.ceil(len(dx) * self.velocity_fraction)))
        mean_dx = dx[-n:].mean() if len(dx) > 0 else 0.0
        mean_dy = dy[-n:].mean() if len(dy) > 0 else 0.0

        recent = ctx.iloc[-(n + 1):]

        # Keep ROT in original deg/min and derive per-step heading increment in degrees
        rot_deg_per_min = np.nanmean(recent['rot'].values)
        if not np.isfinite(rot_deg_per_min):
            rot_deg_per_min = 0.0

        # Fixed sampling interval from preprocessing pipeline
        dt = 30.0

        # Constant turn-rate on displacement vector (direct dx, dy output)
        dpsi_deg = rot_deg_per_min * (dt / 60.0)
        dpsi_rad = np.deg2rad(dpsi_deg)
        cos_dpsi = np.cos(dpsi_rad)
        sin_dpsi = np.sin(dpsi_rad)

        displacements = np.zeros((n_pred_steps, 2), dtype=float)
        cur_dx = float(mean_dx)
        cur_dy = float(mean_dy)

        for i in range(n_pred_steps):
            displacements[i, 0] = cur_dx
            displacements[i, 1] = cur_dy

            next_dx = cur_dx * cos_dpsi - cur_dy * sin_dpsi
            next_dy = cur_dx * sin_dpsi + cur_dy * cos_dpsi
            cur_dx, cur_dy = next_dx, next_dy

        return displacements