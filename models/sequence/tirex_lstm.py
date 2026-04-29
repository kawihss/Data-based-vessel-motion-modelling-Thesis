from __future__ import annotations

import numpy as np
from pathlib import Path

import joblib

from models.base_model import BaselineModel

try:
    from tirex import load_model
except ImportError:  # pragma: no cover
    load_model = None


class TirexLSTMModel(BaselineModel):
    _MODEL_CACHE = {}
    _SCALER_CACHE = {}

    @classmethod
    def clear_cache(cls):
        cls._MODEL_CACHE.clear()
        cls._SCALER_CACHE.clear()
    _DEFAULT_SCALER_PATH = Path(__file__).resolve().parents[2] / "output" / "05_normalized" / "scalers.pkl"

    def __init__(
        self,
        velocity_steps=8,
        model_name="NX-AI/TiRex",
        device=None,
        backend="torch",
        compile_model=False,
        scaler_path=None,
    ):
        super().__init__("TiRexLSTM")
        self.velocity_steps = int(velocity_steps)
        self.model_name = str(model_name)
        self.device = device
        self.backend = backend
        self.compile_model = bool(compile_model)
        self.scaler_path = scaler_path

    def _get_model(self):
        if load_model is None:
            raise ImportError(
                "tirex is not installed. Install with `pip install tirex-ts` and retry."
            )

        cache_key = (self.model_name, self.device, self.backend, self.compile_model)
        model = self._MODEL_CACHE.get(cache_key)
        if model is None:
            model = load_model(
                self.model_name,
                device=self.device,
                backend=self.backend,
                compile=self.compile_model,
            )
            self._MODEL_CACHE[cache_key] = model
        return model

    def _resolve_scaler_path(self):
        if self.scaler_path is None:
            return self._DEFAULT_SCALER_PATH
        return Path(self.scaler_path)

    def _get_dxdy_scaler_params(self):
        scaler_path = self._resolve_scaler_path()
        cache_key = str(scaler_path)
        cached = self._SCALER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if not scaler_path.exists():
            raise RuntimeError(
                f"TiRex scaler not found at '{scaler_path}'. Run 05_normalize.py first."
            )

        scalers = joblib.load(scaler_path)
        global_scaler = scalers.get("global_scaler", {})
        means = np.asarray(global_scaler.get("mean", []), dtype=float)
        scales = np.asarray(global_scaler.get("scale", []), dtype=float)
        if means.size < 2 or scales.size < 2:
            raise RuntimeError(
                f"TiRex scaler at '{scaler_path}' has fewer than 2 features; expected [dx, dy, ...]."
            )

        if not (np.isfinite(scales[0]) and scales[0] > 1e-12):
            raise RuntimeError(f"TiRex scaler dx_scale={scales[0]} is invalid.")
        if not (np.isfinite(scales[1]) and scales[1] > 1e-12):
            raise RuntimeError(f"TiRex scaler dy_scale={scales[1]} is invalid.")

        params = {
            "dx_mean": float(means[0]),
            "dx_scale": float(scales[0]),
            "dy_mean": float(means[1]),
            "dy_scale": float(scales[1]),
        }
        self._SCALER_CACHE[cache_key] = params
        return params

    @staticmethod
    def _normalize_mean_output(mean):
        arr = np.asarray(mean, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr
        if arr.ndim >= 3:
            # Common shapes: (batch, horizon, 1) or (batch, 1, horizon)
            if arr.shape[-1] == 1:
                return arr[..., 0]
            if arr.shape[1] == 1:
                return arr[:, 0, :]
            return arr[:, :, 0]
        return arr

    def _forecast_univariate_channels(self, x_ctx, y_ctx, n_pred_steps):
        model = self._get_model()
        context = np.stack([x_ctx, y_ctx], axis=0).astype(np.float32)
        _, mean = model.forecast(
            context=context,
            output_type="numpy",
            prediction_length=int(n_pred_steps),
        )

        mean_2d = self._normalize_mean_output(mean)
        if mean_2d.ndim == 1:
            raise ValueError(
                f"TiRex returned a 1-D forecast (shape {mean_2d.shape}); expected 2 channels (dx, dy)."
            )

        if mean_2d.shape[0] < 2:
            raise ValueError("TiRex forecast did not return two channel outputs.")

        x_pred = np.asarray(mean_2d[0], dtype=float).reshape(-1)
        y_pred = np.asarray(mean_2d[1], dtype=float).reshape(-1)
        return x_pred[:n_pred_steps], y_pred[:n_pred_steps]

    def _denormalize_displacements(self, pred_norm_dx, pred_norm_dy):
        scaler_params = self._get_dxdy_scaler_params()
        pred_dx = scaler_params["dx_scale"] * np.asarray(pred_norm_dx, dtype=float) + scaler_params["dx_mean"]
        pred_dy = scaler_params["dy_scale"] * np.asarray(pred_norm_dy, dtype=float) + scaler_params["dy_mean"]
        return pred_dx, pred_dy

    def _predict_from_series(self, dx_norm, dy_norm, n_pred_steps):
        if n_pred_steps <= 0:
            return np.empty((0, 2), dtype=float)

        dx_norm = np.asarray(dx_norm, dtype=float).reshape(-1)
        dy_norm = np.asarray(dy_norm, dtype=float).reshape(-1)
        if dx_norm.size == 0 or dy_norm.size == 0:
            return np.zeros((int(n_pred_steps), 2), dtype=float)

        window = max(1, min(int(self.velocity_steps), len(dx_norm), len(dy_norm)))
        norm_dx_ctx = dx_norm[-window:]
        norm_dy_ctx = dy_norm[-window:]

        pred_norm_dx, pred_norm_dy = self._forecast_univariate_channels(
            norm_dx_ctx, norm_dy_ctx, int(n_pred_steps)
        )

        pred_dx, pred_dy = self._denormalize_displacements(pred_norm_dx, pred_norm_dy)
        return np.column_stack([pred_dx, pred_dy])

    def predict_from_cached_track(self, track, n_pred_steps):
        return self._predict_from_series(
            track["dx_norm_ctx"],
            track["dy_norm_ctx"],
            n_pred_steps,
        )

    def predict(self, context_df, n_pred_steps):
        ctx = context_df.sort_values("t_utc") if "t_utc" in context_df.columns else context_df
        if "dx_norm" not in ctx.columns or "dy_norm" not in ctx.columns:
            raise ValueError(
                "TiRexLSTM requires 'dx_norm' and 'dy_norm' columns. Run 05_normalize.py first."
            )
        return self._predict_from_series(
            ctx["dx_norm"].values,
            ctx["dy_norm"].values,
            n_pred_steps,
        )

    def predict_from_arrays(self, x, y, rot, n_pred_steps):
        raise NotImplementedError(
            "TiRexLSTM requires normalized dx/dy inputs. Use predict_from_cached_track or predict instead."
        )
