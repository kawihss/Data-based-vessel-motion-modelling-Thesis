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

        recent_rot = np.asarray(rot[-(n + 1):], dtype=float) # n = 1 number of delas, so n+1 values needed, rot is not delta

        # Keep ROT in original deg/min and derive per-step heading increment in degrees
        rot_deg_per_min = np.nanmean(recent_rot)
        if not np.isfinite(rot_deg_per_min):
            rot_deg_per_min = 0.0

        # Fixed sampling interval from preprocessing pipeline
        dt = 30.0

        # Nautical ROT is clockwise-positive, while math rotation is opposite
        dpsi_deg = -rot_deg_per_min * (dt / 60.0)
        dpsi_rad = np.deg2rad(dpsi_deg)
        alpha = dpsi_rad * dt
        cos_dpsi = np.cos(dpsi_rad)
        sin_dpsi = np.sin(dpsi_rad)

        displacements = np.zeros((n_pred_steps, 2), dtype=float)
        cur_dx = float(mean_dx)
        cur_dy = float(mean_dy)

        for i in range(n_pred_steps):
            #displacements[i, 0] = cur_dx
            #displacements[i, 1] = cur_dy

            #arc-based displacement calculation to avoid accumulating rotation errors over time
            if abs(dpsi_rad) > 1e-6:
                displacements[i, 0] = (cur_dx * np.sin(alpha) + cur_dy * (np.cos(alpha) - 1)) / dpsi_rad
                displacements[i, 1] = (cur_dy * np.sin(alpha) - cur_dx * (np.cos(alpha) - 1)) / dpsi_rad
            else:
                displacements[i, 0] = cur_dx * dt
                displacements[i, 1] = cur_dy * dt

            # Rotate the velocity vector for the next step
            next_dx = cur_dx * cos_dpsi - cur_dy * sin_dpsi
            next_dy = cur_dx * sin_dpsi + cur_dy * cos_dpsi
            cur_dx, cur_dy = next_dx, next_dy

        return displacements


class HybridCVCTRVModel(BaselineModel):
    def __init__(self, cv_velocity_steps=1, ctrv_velocity_steps=1, rot_steps=1, rot_threshold=1.0):
        super().__init__("HybridCVCTRV")
        self.cv_velocity_steps = cv_velocity_steps
        self.ctrv_velocity_steps = ctrv_velocity_steps
        self.rot_steps = rot_steps
        self.rot_threshold = rot_threshold

    def predict(self, context_df, n_pred_steps):
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values
        if 'rot' not in ctx.columns:
            raise ValueError("HybridCVCTRVModel requires a 'rot' column in the context data.")
        rot = ctx['rot'].values
        return self.predict_from_arrays(x, y, rot, n_pred_steps)

    def _window_steps(self, velocity_steps, n_points):
        if n_points <= 1:
            raise ValueError("HybridCVCTRVModel requires at least two context points.")
        n_deltas = n_points - 1
        if int(velocity_steps) < 1:
            raise ValueError("HybridCVCTRVModel requires velocity_steps >= 1.")
        return max(1, min(n_deltas, int(velocity_steps)))

    def predict_from_arrays(self, x, y, rot, n_pred_steps):
        if n_pred_steps <= 0:
            return np.empty((0, 2), dtype=float)

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if rot is None:
            raise ValueError("HybridCVCTRVModel requires rot values for CTRV switching.")
        rot_values = np.asarray(rot, dtype=float)

        cv_velocity_steps = self._window_steps(self.cv_velocity_steps, len(x))
        ctrv_velocity_steps = self._window_steps(self.ctrv_velocity_steps, len(x))
        n = max(1, min(len(rot_values), int(self.rot_steps))) #clipping

        rot_recent = np.abs(rot_values[-n:])
        mean_abs_rot = float(np.nanmean(rot_recent)) if len(rot_recent) > 0 else 0.0
        if not np.isfinite(mean_abs_rot):
            mean_abs_rot = 0.0

        if mean_abs_rot >= float(self.rot_threshold):
            model = ConstantTurnRateVelocityModel(velocity_steps=ctrv_velocity_steps)
        else:
            model = ConstantVelocityModel(velocity_steps=cv_velocity_steps)

        return model.predict_from_arrays(x, y, rot_values, n_pred_steps)