import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

from .base_model import BaselineModel

# Implements KF with the state vector 
# (x, y, dx, dy) and constant velocity motion model

import numpy as np
from .base_model import BaselineModel


# Implements KF with the state vector 
# (x, y, dx, dy) and constant velocity motion model


class KalmanFilter(BaselineModel):
    def __init__(# default values tuned on small subset of validation data
        self,
        q_pos=0.08935110590331861,
        q_vel=0.2385816677142884,
        r_pos=15.409457762881532,
        p0_pos=2105.1180519608724,
        p0_vel=0.20207122587167334,
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

        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)
        self.Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(float)
        self.R = np.diag([self.r_pos, self.r_pos]).astype(float)

    def predict(self, context_df, n_pred_steps):
        ctx = context_df.sort_values('t_utc')
        x = ctx['x'].values
        y = ctx['y'].values
        rot = ctx['rot'].values if 'rot' in ctx.columns else None # rot not used, consistency
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

        state = kf["x"]
        P = kf["P"]

        # Context phase: regular predict-update cycles with measurements.
        # used to learn internal state and covariance, not for prediction
        for i in range(len(x_valid)):
            z = np.array([x_valid[i], y_valid[i]], dtype=float)
            state, P = self._kf_predict(state, P) #predict next state based on model
            state, P = self._kf_update(state, P, z) #correct with measurement, adjust internal state and covariance
            # bail out if state or covariance has diverged
            if not np.all(np.isfinite(state)) or np.trace(P) > 1e12:
                return np.zeros((n_pred_steps, 2), dtype=float)

        # Prediction phase: pure rollout, no measurement corrections.
        displacements = np.zeros((n_pred_steps, 2), dtype=float)
        prev_x, prev_y = float(state[0]), float(state[1])

        for i in range(n_pred_steps):
            state, P = self._kf_predict(state, P) #predict next state based on model
            cur_x, cur_y = float(state[0]), float(state[1]) #read predicted position
            if not (np.isfinite(cur_x) and np.isfinite(cur_y)) or np.trace(P) > 1e12:
                break
            displacements[i, 0] = cur_x - prev_x
            displacements[i, 1] = cur_y - prev_y
            prev_x, prev_y = cur_x, cur_y

        return displacements

    def _build_filter(self, x0, y0, v0x, v0y):
        #see thesis text for matrix definitions
        x = np.array([x0, y0, v0x, v0y], dtype=float)
        P = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel]).astype(float)
        return {"x": x, "P": P}

    def _kf_predict(self, state, P):
        state = self.F @ state
        P = self.F @ P @ self.F.T + self.Q
        P = 0.5 * (P + P.T)
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-9))
        return state, P

    def _kf_update(self, state, P, z):
        S = self.H @ P @ self.H.T + self.R
        S = 0.5 * (S + S.T)

        PHt = P @ self.H.T
        eps = 1e-9
        try:
            K = np.linalg.solve(S + eps * np.eye(S.shape[0]), PHt.T).T
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S)

        state = state + K @ (z - self.H @ state)
        I = np.eye(4)
        # Joseph form improves numerical stability and keeps P positive semi-definite.
        P = (I - K @ self.H) @ P @ (I - K @ self.H).T + K @ self.R @ K.T
        P = 0.5 * (P + P.T)
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-9))
        return state, P

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

class CTRVExtendedKalmanFilter(BaselineModel):
	# Similar to KalmanFilter but with CTRV motion model and state vector (x, y, dx, dy, dpsi).
	# not using library 
	def __init__( # results of tuning on small subset of validation data
		self,
		q_pos=4.12068400231886,
		q_vel=0.005776275613820111,
		q_rot=1.0020484272148286e-07,
		r_pos=0.2208390262563217,
		p0_pos=0.5359918958261461,
		p0_vel=597.7148765174895,
		p0_rot=0.0021661420901010215,
		dt=30.0,
		init_velocity_steps=3,
	):
		super().__init__("CTRVExtendedKalman")
		self.q_pos = float(q_pos)
		self.q_vel = float(q_vel)
		self.q_rot = float(q_rot)
		self.r_pos = float(r_pos)
		self.p0_pos = float(p0_pos)
		self.p0_vel = float(p0_vel)
		self.p0_rot = float(p0_rot)
		self.dt = float(dt)
		self.init_velocity_steps = int(init_velocity_steps)

		self.Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_rot]).astype(float)
		self.R = np.diag([self.r_pos, self.r_pos]).astype(float)
		self.C = np.array([
			[1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0],
		], dtype=float)

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

		rot_valid = None
		if rot is not None:
			rot_arr = np.asarray(rot, dtype=float)
			if len(rot_arr) == len(valid_mask):
				rot_valid = rot_arr[valid_mask]

		v0x, v0y = self._estimate_initial_velocity(x_valid, y_valid)
		# Limit context to the most recent init_velocity_steps points
		n_ctx = max(1, int(self.init_velocity_steps))
		x_ctx = x_valid[-n_ctx:]
		y_ctx = y_valid[-n_ctx:]
		rot_ctx = rot_valid[-n_ctx:] if rot_valid is not None else None

		ekf = self._build_filter(x_ctx[0], y_ctx[0], v0x, v0y, dpsi0=0.0)

		state = ekf["x"]
		P = ekf["P"]

		# Context phase: predict-update cycles with position measurements
		# used to learn internal state and covariance, not for prediction
		for i in range(len(x_ctx)):
			z = np.array([x_ctx[i], y_ctx[i]], dtype=float)
			state, P = self._ekf_predict(state, P)
			state, P = self._ekf_update(state, P, z)
			# bail out if state or covariance has diverged
			if not np.all(np.isfinite(state)) or np.trace(P) > 1e12:
				return np.zeros((n_pred_steps, 2), dtype=float)

		# Prediction phase: pure rollout without measurement corrections
		displacements = np.zeros((n_pred_steps, 2), dtype=float)
		prev_x, prev_y = float(state[0]), float(state[1])

		for i in range(n_pred_steps):
			state, P = self._ekf_predict(state, P)
			cur_x, cur_y = float(state[0]), float(state[1])
			if not (np.isfinite(cur_x) and np.isfinite(cur_y)) or np.trace(P) > 1e12:
				break
			displacements[i, 0] = cur_x - prev_x
			displacements[i, 1] = cur_y - prev_y
			prev_x, prev_y = cur_x, cur_y

		return displacements

	def _build_filter(self, x0, y0, v0x, v0y, dpsi0):
		x = np.array([x0, y0, v0x, v0y, dpsi0], dtype=float)
		P = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel, self.p0_rot]).astype(float)
		return {"x": x, "P": P}

	def _f(self, state):
		x, y, dx, dy, dpsi = state
		alpha = dpsi * self.dt
		cos_a = np.cos(alpha)
		sin_a = np.sin(alpha)

		if abs(dpsi) > 1e-6:
			new_x = x + (dx * sin_a + dy * (cos_a - 1.0)) / dpsi
			new_y = y + (dy * sin_a - dx * (cos_a - 1.0)) / dpsi
		else:
			new_x = x + dx * self.dt
			new_y = y + dy * self.dt

		new_dx = dx * cos_a - dy * sin_a
		new_dy = dx * sin_a + dy * cos_a
		return np.array([
			new_x,
			new_y,
			new_dx,
			new_dy,
			dpsi,
		], dtype=float)

	def _jacobian(self, state):
		_, _, dx, dy, dpsi = state
		alpha = dpsi * self.dt
		cos_a = np.cos(alpha)
		sin_a = np.sin(alpha)
		T = self.dt

		if abs(dpsi) > 1e-6:
			dx_ddpsi = (
				dpsi * T * (dx * cos_a - dy * sin_a)
				- (dx * sin_a + dy * (cos_a - 1.0))
			) / (dpsi ** 2)
			dy_ddpsi = (
				dpsi * T * (dy * cos_a + dx * sin_a)
				- (dy * sin_a - dx * (cos_a - 1.0))
			) / (dpsi ** 2)

			A = np.array([
				[1.0, 0.0, sin_a / dpsi, (cos_a - 1.0) / dpsi, dx_ddpsi],
				[0.0, 1.0, (1.0 - cos_a) / dpsi, sin_a / dpsi, dy_ddpsi],
				[0.0, 0.0, cos_a, -sin_a, (-dx * sin_a - dy * cos_a) * T],
				[0.0, 0.0, sin_a, cos_a, (dx * cos_a - dy * sin_a) * T],
				[0.0, 0.0, 0.0, 0.0, 1.0],
			], dtype=float)
		else:
			A = np.array([
				[1.0, 0.0, T, 0.0, -0.5 * dy * (T ** 2)],
				[0.0, 1.0, 0.0, T, 0.5 * dx * (T ** 2)],
				[0.0, 0.0, 1.0, 0.0, -dy * T],
				[0.0, 0.0, 0.0, 1.0, dx * T],
				[0.0, 0.0, 0.0, 0.0, 1.0],
			], dtype=float)
		return A

	def _ekf_predict(self, state, P):
		A = self._jacobian(state)
		state = self._f(state)
		P = A @ P @ A.T + self.Q
		P = 0.5 * (P + P.T)
		np.fill_diagonal(P, np.maximum(np.diag(P), 1e-9))
		return state, P

	def _ekf_update(self, state, P, z):
		S = self.C @ P @ self.C.T + self.R
		S = 0.5 * (S + S.T)

		PCt = P @ self.C.T
		eps = 1e-9
		try:
			L = np.linalg.solve(S + eps * np.eye(S.shape[0]), PCt.T).T
		except np.linalg.LinAlgError:
			L = PCt @ np.linalg.pinv(S)

		state = state + L @ (z - self.C @ state)
		I = np.eye(5)
		# Joseph form improves numerical stability and keeps P positive semi-definite.
		P = (I - L @ self.C) @ P @ (I - L @ self.C).T + L @ self.R @ L.T
		P = 0.5 * (P + P.T)
		np.fill_diagonal(P, np.maximum(np.diag(P), 1e-9))
		return state, P

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

