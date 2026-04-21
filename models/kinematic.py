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