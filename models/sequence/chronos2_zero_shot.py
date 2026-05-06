from pathlib import Path

import numpy as np
import joblib
import torch
from chronos import Chronos2Pipeline

from models.base_model import BaselineModel


_CONTEXT_LENGTH = 10
_PREDICTION_LENGTH = 10
_REQUIRED_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_COVARIATE_COLUMNS = ("dx_norm", "dy_norm", "sog_norm", "cog_sin_norm", "cog_cos_norm", "dt_norm", "rot_norm")


class Chronos2ZeroShotModel(BaselineModel):
    _PIPELINE_CACHE = {}
    _SCALER_CACHE = {}

    def __init__(self, model_name, device_map, max_memory, torch_dtype, scaler_path):
        super().__init__("Chronos-2 Zero-Shot")
        self.model_name = model_name
        self.device_map = device_map
        self.max_memory = max_memory
        self.torch_dtype = torch_dtype
        self.scaler_path = Path(scaler_path)

    def _get_scaler_params(self):
        cached = self._SCALER_CACHE.get(self.scaler_path)
        if cached is not None:
            return cached
        scalers = joblib.load(self.scaler_path)
        gs = scalers["global_scaler"]
        means = np.asarray(gs["mean"], dtype=float)
        scales = np.asarray(gs["scale"], dtype=float)
        params = {
            "dx_mean": float(means[0]), "dx_scale": float(scales[0]),
            "dy_mean": float(means[1]), "dy_scale": float(scales[1]),
        }
        self._SCALER_CACHE[self.scaler_path] = params
        return params

    def _denormalize(self, norm_dx, norm_dy):
        p = self._get_scaler_params()
        return (
            p["dx_scale"] * np.asarray(norm_dx, dtype=float) + p["dx_mean"],
            p["dy_scale"] * np.asarray(norm_dy, dtype=float) + p["dy_mean"],
        )

    def _get_pipeline(self):
        cache_key = (self.model_name, self.device_map, self.torch_dtype)
        if cache_key not in self._PIPELINE_CACHE:
            self._PIPELINE_CACHE[cache_key] = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                dtype=getattr(torch, self.torch_dtype),
                max_memory=self.max_memory,
            )
        return self._PIPELINE_CACHE[cache_key]

    def _build_input(self, context_df):
        ctx = context_df.sort_values("t_utc").tail(_CONTEXT_LENGTH)
        target = np.vstack([
            ctx["dx_norm"].to_numpy(dtype=np.float32, copy=True),
            ctx["dy_norm"].to_numpy(dtype=np.float32, copy=True),
        ])
        past_covariates = {col: ctx[col].to_numpy(dtype=np.float32, copy=True) for col in _COVARIATE_COLUMNS}
        return {"target": target, "past_covariates": past_covariates}

    def predict_quantiles(self, context_df, n_pred_steps):
        pipeline = self._get_pipeline()
        quantile_levels = tuple(round(float(q), 2) for q in pipeline.model.quantiles.detach().cpu().tolist())
        forecast = pipeline.predict(
            inputs=[self._build_input(context_df)],
            prediction_length=_PREDICTION_LENGTH,
            batch_size=1,
            context_length=_CONTEXT_LENGTH,
        )[0].detach().cpu().numpy()  # shape: (2, n_quantiles, pred_len)

        result = {}
        for i, level in enumerate(quantile_levels):
            pred_dx, pred_dy = self._denormalize(forecast[0, i, :], forecast[1, i, :])
            result[float(level)] = np.column_stack([pred_dx, pred_dy])
        return result

    def predict(self, context_df, n_pred_steps):
        return np.asarray(self.predict_quantiles(context_df, n_pred_steps)[0.5], dtype=float)