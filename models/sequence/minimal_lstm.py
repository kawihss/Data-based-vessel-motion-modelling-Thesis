from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from models.base_model import BaselineModel

CONTEXT_LEN = 10
PRED_LEN = 10
FEATURE_COLUMNS = ("dx_norm", "dy_norm")
TARGET_COLUMNS = ("dx_norm", "dy_norm")


class MinimalLSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, pred_len * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [batch, time, features], got shape {tuple(x.shape)}")
        _, (hidden_n, _) = self.lstm(x)
        return self.head(hidden_n[-1]).reshape(-1, self.pred_len, 2)


class MinimalLSTMModel(BaselineModel):
    def __init__(self, checkpoint_path: str, scaler_path: str, device: str):
        super().__init__("MinimalLSTM")
        self.device = torch.device(device)

        ckpt = torch.load(Path(checkpoint_path), map_location=self.device)
        for key in ("state_dict", "pred_len", "input_size", "hidden_size", "num_layers", "dropout"):
            if key not in ckpt:
                raise KeyError(f"Checkpoint missing required key: {key}")

        self.pred_len = int(ckpt["pred_len"])
        self.model = MinimalLSTMNet(
            input_size=int(ckpt["input_size"]),
            hidden_size=int(ckpt["hidden_size"]),
            num_layers=int(ckpt["num_layers"]),
            dropout=float(ckpt["dropout"]),
            pred_len=self.pred_len,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

        scalers = joblib.load(Path(scaler_path))
        gs = scalers["global_scaler"]
        means = np.asarray(gs["mean"], dtype=float)
        scales = np.asarray(gs["scale"], dtype=float)
        self.dx_mean, self.dx_scale = float(means[0]), float(scales[0])
        self.dy_mean, self.dy_scale = float(means[1]), float(scales[1])

    def predict(self, context_df, n_pred_steps):
        if int(n_pred_steps) != self.pred_len:
            raise ValueError(f"n_pred_steps must equal checkpoint pred_len={self.pred_len}, got {n_pred_steps}")
        if "t_utc" in context_df.columns:
            context_df = context_df.sort_values("t_utc")
        if len(context_df) < CONTEXT_LEN:
            raise ValueError(f"Context too short: {len(context_df)} < {CONTEXT_LEN}")
        x = torch.from_numpy(
            context_df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-CONTEXT_LEN:]
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(x).squeeze(0).cpu().numpy().astype(float)
        dx = pred_norm[:, 0] * self.dx_scale + self.dx_mean
        dy = pred_norm[:, 1] * self.dy_scale + self.dy_mean
        return np.column_stack([dx, dy])
