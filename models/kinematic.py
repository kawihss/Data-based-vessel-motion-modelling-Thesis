import numpy as np
from .base_model import BaselineModel


class ConstantVelocityModel(BaselineModel):
    def __init__(self, velocity_steps=1):
        super().__init__("ConstantVelocity")
        # Number of recent context steps used for velocity averaging.
        self.velocity_steps = velocity_steps

    def predict(self, context_df, n_pred_steps):
        # Compute dx, dy from raw x, y positions sorted by time
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values

        return self.predict_from_arrays(x, y, None, n_pred_steps)

    def predict_from_arrays(self, x, y, rot, n_pred_steps):
        if n_pred_steps <= 0:
            return np.empty((0, 2), dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)

        n = max(1, min(len(dx), int(self.velocity_steps)))
        mean_dx = dx[-n:].mean() if n > 0 else 0.0
        mean_dy = dy[-n:].mean() if n > 0 else 0.0

        # Repeat constant displacement for every future step
        # Returns shape (n_pred_steps, 2) in meters
        return np.tile([mean_dx, mean_dy], (n_pred_steps, 1))


class ConstantTurnRateVelocityModel(BaselineModel):
    def __init__(self, velocity_steps=1):
        super().__init__("CTRV")
        # Number of recent context steps used to estimate displacement and turn rate.
        self.velocity_steps = velocity_steps

    def predict(self, context_df, n_pred_steps):
        # Estimate state from raw context sorted by time
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values
        rot = ctx['rot'].values

        return self.predict_from_arrays(x, y, rot, n_pred_steps)

    def predict_from_arrays(self, x, y, rot, n_pred_steps):
        if n_pred_steps <= 0:
            return np.empty((0, 2), dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        n = max(1, min(len(dx), int(self.velocity_steps)))
        mean_dx = dx[-n:].mean() if len(dx) > 0 else 0.0
        mean_dy = dy[-n:].mean() if len(dy) > 0 else 0.0

        recent_rot = np.asarray(rot[-(n + 1):], dtype=float)

        # Keep ROT in original deg/min and derive per-step heading increment in degrees
        rot_deg_per_min = np.nanmean(recent_rot)
        if not np.isfinite(rot_deg_per_min):
            rot_deg_per_min = 0.0

        # Fixed sampling interval from preprocessing pipeline
        dt = 30.0

        # Nautical ROT is clockwise-positive, while math rotation is CCW-positive.
        # Negating aligns turn direction with the XY rotation matrix below.
        dpsi_deg = -rot_deg_per_min * (dt / 60.0)
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