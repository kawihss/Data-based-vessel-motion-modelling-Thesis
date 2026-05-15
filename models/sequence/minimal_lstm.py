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
DOMAIN_ONE_HOT = {
    "river": (1.0, 0.0, 0.0, 0.0),
    "harbour": (0.0, 1.0, 0.0, 0.0),
    "channel": (0.0, 0.0, 1.0, 0.0),
    "lock": (0.0, 0.0, 0.0, 1.0),
}
DOMAIN_LABELS = tuple(DOMAIN_ONE_HOT.keys())
DOMAIN_TO_INDEX = {name: idx for idx, name in enumerate(DOMAIN_LABELS)}
DOMAIN_ONE_HOT_TENSOR = torch.tensor(list(DOMAIN_ONE_HOT.values()), dtype=torch.float32)

#called "MinimalLSTM" because it's a very minimal implementation of an LSTM-based model,
# no autoregresive decoding, linear head

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

    def forward(self, x: torch.Tensor) -> torch.Tensor: # forward pass
        #hidden_n[-1] is the last layer's hidden state 
        # fullk horizon prediction is made in one shot from the last hidden state via the linear head; no autoregressive decoding
        _, (hidden_n, _) = self.lstm(x)
        return self.head(hidden_n[-1]).reshape(-1, self.pred_len, 2)


class MinimalLSTMDomainNet(MinimalLSTMNet):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, pred_len: int):
        self.base_hidden_size = hidden_size
        self.domain_size = 4
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size + self.domain_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
        )
    def _build_initial_state(self, x: torch.Tensor, domain_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size

        domain_ids = domain_idx.to(device=x.device, dtype=torch.long)
        if domain_ids.dim() == 0:
            domain_ids = domain_ids.view(1).repeat(batch_size)
        one_hot = DOMAIN_ONE_HOT_TENSOR.to(device=x.device, dtype=x.dtype)[domain_ids]
        one_hot = one_hot.unsqueeze(0).expand(num_layers, -1, -1)

        h0_base = torch.zeros(
            (num_layers, batch_size, hidden_size - self.domain_size),
            dtype=x.dtype,
            device=x.device,
        )
        c0_base = torch.zeros(
            (num_layers, batch_size, hidden_size - self.domain_size),
            dtype=x.dtype,
            device=x.device,
        )
        h0 = torch.cat([h0_base, one_hot], dim=-1)
        c0 = torch.cat([c0_base, one_hot], dim=-1)
        return h0, c0

    def forward(self, x: torch.Tensor, domain_idx: torch.Tensor) -> torch.Tensor:
        # Domain is injected only once via initial hidden/cell states.
        h0, c0 = self._build_initial_state(x, domain_idx)
        _, (hidden_n, _) = self.lstm(x, (h0, c0))
        return self.head(hidden_n[-1]).reshape(-1, self.pred_len, 2)


class MinimalLSTMModel(BaselineModel):
    def __init__(self, checkpoint_path: str, scaler_path: str, device: str):
        super().__init__("MinimalLSTM")
        self.device = torch.device(device)

        ckpt = torch.load(Path(checkpoint_path), map_location=self.device) # load trained model checkpoint (to pass it to run_evaluation.py)
     
        self.pred_len = int(ckpt["pred_len"])
        self.feature_columns = tuple(ckpt.get("feature_columns", FEATURE_COLUMNS))
        self.model = MinimalLSTMNet(
            input_size=int(ckpt["input_size"]),
            hidden_size=int(ckpt["hidden_size"]),
            num_layers=int(ckpt["num_layers"]),
            dropout=float(ckpt["dropout"]),
            pred_len=self.pred_len,
        ).to(self.device)
        self.model.load_state_dict(self._normalize_state_dict_keys(ckpt["state_dict"]), strict=True)
        self.model.eval()

        scalers = joblib.load(Path(scaler_path))
        gs = scalers["global_scaler"]
        means = np.asarray(gs["mean"], dtype=float)
        scales = np.asarray(gs["scale"], dtype=float)
        self.dx_mean, self.dx_scale = float(means[0]), float(scales[0])
        self.dy_mean, self.dy_scale = float(means[1]), float(scales[1])

    @staticmethod
    def _normalize_state_dict_keys(state_dict):
        #remove wrappers introduced by torch.compile and DataParallel to ensure compatibility when loading the state dict
        normalized = {}
        for key, value in state_dict.items():
            new_key = key
            # Remove wrappers introduced by torch.compile and DataParallel.
            while new_key.startswith("_orig_mod.") or new_key.startswith("module."):
                if new_key.startswith("_orig_mod."):
                    new_key = new_key[len("_orig_mod."):]
                elif new_key.startswith("module."):
                    new_key = new_key[len("module."):]
            normalized[new_key] = value
        return normalized

    def predict(self, context_df, n_pred_steps):
        #prepare input tensor from context_df, run through model, and denormalize the predictions to return in original scale
        x = torch.from_numpy(
            context_df.loc[:, self.feature_columns].to_numpy(dtype=np.float32)[-CONTEXT_LEN:]
        ).unsqueeze(0).to(self.device) 
        with torch.no_grad():
            pred_norm = self.model(x).squeeze(0).cpu().numpy().astype(float)
        dx = pred_norm[:, 0] * self.dx_scale + self.dx_mean
        dy = pred_norm[:, 1] * self.dy_scale + self.dy_mean
        return np.column_stack([dx, dy])


class MinimalLSTMDomainModel(BaselineModel):
    def __init__(self, checkpoint_path: str, scaler_path: str, device: str):
        super().__init__("MinimalLSTMDomain")
        self.device = torch.device(device)

        ckpt = torch.load(Path(checkpoint_path), map_location=self.device)

        self.pred_len = int(ckpt["pred_len"])
        self.feature_columns = tuple(ckpt.get("feature_columns", FEATURE_COLUMNS))
        self.model = MinimalLSTMDomainNet(
            input_size=int(ckpt["input_size"]),
            hidden_size=int(ckpt["hidden_size"]),
            num_layers=int(ckpt["num_layers"]),
            dropout=float(ckpt["dropout"]),
            pred_len=self.pred_len,
        ).to(self.device)
        self.model.load_state_dict(MinimalLSTMModel._normalize_state_dict_keys(ckpt["state_dict"]), strict=True)
        self.model.eval()

        scalers = joblib.load(Path(scaler_path))
        gs = scalers["global_scaler"]
        means = np.asarray(gs["mean"], dtype=float)
        scales = np.asarray(gs["scale"], dtype=float)
        self.dx_mean, self.dx_scale = float(means[0]), float(scales[0])
        self.dy_mean, self.dy_scale = float(means[1]), float(scales[1])

    def predict(self, context_df, n_pred_steps):
        x = torch.from_numpy(
            context_df.loc[:, self.feature_columns].to_numpy(dtype=np.float32)[-CONTEXT_LEN:]
        ).unsqueeze(0).to(self.device)
        domain_idx = torch.tensor([DOMAIN_TO_INDEX[str(context_df["context"].iloc[-1])]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred_norm = self.model(x, domain_idx).squeeze(0).cpu().numpy().astype(float)
        dx = pred_norm[:, 0] * self.dx_scale + self.dx_mean
        dy = pred_norm[:, 1] * self.dy_scale + self.dy_mean
        return np.column_stack([dx, dy])
