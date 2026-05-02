from __future__ import annotations

from pathlib import Path

import numpy as np

from models.base_model import BaselineModel

try:
    from chronos import Chronos2Pipeline
except ImportError:  # pragma: no cover
    Chronos2Pipeline = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Chronos2ZeroShotModel(BaselineModel):
    _PIPELINE_CACHE = {}
    _REQUIRED_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
    _REQUIRED_COLUMNS = (
        "dx",
        "dy",
        "dx_norm",
        "dy_norm",
        "sog_norm",
        "cog_sin_norm",
        "cog_cos_norm",
        "dt_norm",
        "rot_norm",
    )

    def __init__(
        self,
        model_name="amazon/chronos-2",
        device_map=None,
        torch_dtype=None,
        batch_size=1,
        context_length=10,
        prediction_length=10,
    ):
        super().__init__("Chronos-2 Zero-Shot")
        self.model_name = str(model_name)
        self.device_map = None if device_map is None else str(device_map)
        self.torch_dtype = None if torch_dtype is None else str(torch_dtype)
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)

        if self.context_length != 10:
            raise ValueError(f"Chronos-2 requires context_length=10, got {self.context_length}.")
        if self.prediction_length != 10:
            raise ValueError(f"Chronos-2 requires prediction_length=10, got {self.prediction_length}.")

    @classmethod
    def clear_cache(cls):
        cls._PIPELINE_CACHE.clear()

    def _get_pipeline(self):
       
        cache_key = (self.model_name, self.device_map, self.torch_dtype)
        pipeline = self._PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            load_kwargs = {}
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map
            if self.torch_dtype is not None:
                if torch is None:
                    raise ImportError("torch is required when torch_dtype is set for Chronos-2.")
                load_kwargs["dtype"] = getattr(torch, self.torch_dtype)
            pipeline = Chronos2Pipeline.from_pretrained(self.model_name, **load_kwargs)
            self._PIPELINE_CACHE[cache_key] = pipeline
        return pipeline

    def _get_quantile_levels(self, pipeline):
        quantiles = tuple(round(float(value), 2) for value in pipeline.model.quantiles.detach().cpu().tolist())
        missing = [quantile for quantile in self._REQUIRED_QUANTILES if quantile not in quantiles]
        if missing:
            raise ValueError(f"Chronos-2 native quantile grid is missing required quantiles: {missing}.")
        return quantiles

    def _validate_context(self, context_df, n_pred_steps):
        missing = [column for column in self._REQUIRED_COLUMNS if column not in context_df.columns]
        if missing:
            raise ValueError(
                "Chronos-2 requires columns: " + ", ".join(self._REQUIRED_COLUMNS) + f". Missing: {missing}."
            )
        if int(n_pred_steps) != self.prediction_length:
            raise ValueError(
                f"Chronos-2 requires prediction_length={self.prediction_length}, got {int(n_pred_steps)}."
            )

    def _build_input(self, context_df):
        ctx = context_df.sort_values("t_utc") if "t_utc" in context_df.columns else context_df
        ctx = ctx.tail(self.context_length)
        if len(ctx) != self.context_length:
            raise ValueError(
                f"Chronos-2 requires exactly {self.context_length} context steps, got {len(ctx)}."
            )

        target = np.vstack([
            ctx["dx"].to_numpy(dtype=np.float32, copy=True),
            ctx["dy"].to_numpy(dtype=np.float32, copy=True),
        ])
        past_covariates = {
            "dx_norm": ctx["dx_norm"].to_numpy(dtype=np.float32, copy=True),
            "dy_norm": ctx["dy_norm"].to_numpy(dtype=np.float32, copy=True),
            "sog_norm": ctx["sog_norm"].to_numpy(dtype=np.float32, copy=True),
            "cog_sin_norm": ctx["cog_sin_norm"].to_numpy(dtype=np.float32, copy=True),
            "cog_cos_norm": ctx["cog_cos_norm"].to_numpy(dtype=np.float32, copy=True),
            "dt_norm": ctx["dt_norm"].to_numpy(dtype=np.float32, copy=True),
            "rot_norm": ctx["rot_norm"].to_numpy(dtype=np.float32, copy=True),
        }
        return {"target": target, "past_covariates": past_covariates}

    def _predict_quantile_tensor(self, context_df, n_pred_steps):
        self._validate_context(context_df, n_pred_steps)
        pipeline = self._get_pipeline()
        quantile_levels = self._get_quantile_levels(pipeline)
        forecast = pipeline.predict(
            inputs=[self._build_input(context_df)],
            prediction_length=self.prediction_length,
            batch_size=self.batch_size,
            context_length=self.context_length,
        )[0]
        forecast_np = forecast.detach().cpu().numpy() if hasattr(forecast, "detach") else np.asarray(forecast)
        if forecast_np.shape != (2, len(quantile_levels), self.prediction_length):
            raise ValueError(
                f"Unexpected Chronos-2 forecast shape {forecast_np.shape}; expected (2, {len(quantile_levels)}, {self.prediction_length})."
            )
        return forecast_np, quantile_levels

    def predict_quantiles(self, context_df, n_pred_steps):
        forecast_np, quantile_levels = self._predict_quantile_tensor(context_df, n_pred_steps)
        return {
            float(level): np.column_stack([forecast_np[0, index, :], forecast_np[1, index, :]])
            for index, level in enumerate(quantile_levels)
        }

    def predict(self, context_df, n_pred_steps):
        quantiles = self.predict_quantiles(context_df, n_pred_steps)
        return np.asarray(quantiles[0.5], dtype=float)

    def predict_from_arrays(self, x, y, rot, n_pred_steps):
        raise NotImplementedError(
            "Chronos-2 requires DataFrame inputs with dx/dy targets and normalized covariates. Use predict()."
        )