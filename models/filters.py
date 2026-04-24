import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

from .base_model import BaselineModel


class KalmanFilter(BaselineModel):
	def __init__(
		self,
		q_pos=1.0,
		q_vel=0.1,
		r_pos=25.0,
		p0_pos=100.0,
		p0_vel=10.0,
		dt=30.0,
		init_velocity_steps=3,
	):
		super().__init__("Kalman")
		self.q_pos = float(q_pos)
		self.q_vel = float(q_vel)
		self.r_pos = float(r_pos)
		self.p0_pos = float(p0_pos)
		self.p0_vel = float(p0_vel)
		self.dt = float(dt)
		self.init_velocity_steps = int(init_velocity_steps)

	def predict(self, context_df, n_pred_steps):
		ctx = context_df.sort_values('t_utc')
		x = ctx['x'].values
		y = ctx['y'].values
		rot = ctx['rot'].values if 'rot' in ctx.columns else None
		return self.predict_from_arrays(x, y, rot, n_pred_steps)

	def predict_from_arrays(self, x, y, rot, n_pred_steps):
		if n_pred_steps <= 0:
			return np.empty((0, 2), dtype=float)

		x = np.asarray(x, dtype=float)
		y = np.asarray(y, dtype=float)

		valid_mask = np.isfinite(x) & np.isfinite(y)
		if not np.any(valid_mask):
			return np.zeros((n_pred_steps, 2), dtype=float)

		x_valid = x[valid_mask]
		y_valid = y[valid_mask]
		if len(x_valid) == 1:
			return np.zeros((n_pred_steps, 2), dtype=float)

		v0x, v0y = self._estimate_initial_velocity(x_valid, y_valid)
		kf = self._build_filter(x_valid[0], y_valid[0], v0x, v0y)

		# Context phase: regular predict-update cycles with measurements.
		for i in range(len(x_valid)):
			z = np.array([x_valid[i], y_valid[i]], dtype=float)
			kf.predict()
			kf.update(z)

		# Prediction phase: pure rollout, no measurement corrections.
		displacements = np.zeros((n_pred_steps, 2), dtype=float)
		prev_x, prev_y = float(kf.x[0, 0]), float(kf.x[1, 0])

		for i in range(n_pred_steps):
			kf.predict()
			cur_x, cur_y = float(kf.x[0, 0]), float(kf.x[1, 0])
			displacements[i, 0] = cur_x - prev_x
			displacements[i, 1] = cur_y - prev_y
			prev_x, prev_y = cur_x, cur_y

		return displacements

	def _build_filter(self, x0, y0, v0x, v0y):
		kf = FilterPyKalmanFilter(dim_x=4, dim_z=2)
		kf.x = np.array([[x0], [y0], [v0x], [v0y]], dtype=float)

		kf.F = np.array([
			[1.0, 0.0, self.dt, 0.0],
			[0.0, 1.0, 0.0, self.dt],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		], dtype=float)
		kf.H = np.array([
			[1.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0],
		], dtype=float)
		kf.Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(float)
		kf.R = np.diag([self.r_pos, self.r_pos]).astype(float)
		kf.P = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel]).astype(float)
		return kf

	def _estimate_initial_velocity(self, x, y):
		dx = np.diff(x)
		dy = np.diff(y)
		if len(dx) == 0:
			return 0.0, 0.0

		n = max(1, min(len(dx), self.init_velocity_steps))
		mean_dx = float(np.nanmean(dx[-n:]))
		mean_dy = float(np.nanmean(dy[-n:]))
		if not np.isfinite(mean_dx):
			mean_dx = 0.0
		if not np.isfinite(mean_dy):
			mean_dy = 0.0

		return mean_dx / self.dt, mean_dy / self.dt