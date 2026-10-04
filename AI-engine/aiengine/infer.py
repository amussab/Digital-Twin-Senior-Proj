"""Runtime tensor path shared by evaluate.py, export.py and engine.py.

The deployed host (C#) builds two float tensors per model call and nothing else:
    encoder_cont  float32[B, L, R]   scaled inputs of the L previous windows
    decoder_cont  float32[B, D, R]   scaled inputs of the D decoder windows
Scaling is (x - center) / scale with the per-input constants in model_contract.json.
ExportWrapper rebuilds the rest of pytorch-forecasting's input dict from fixed lengths, so the
ONNX graph and the PyTorch path consume exactly the same two tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, encoder_length: int, decoder_length: int,
                 target_scale: np.ndarray):
        super().__init__()
        self.model = model
        self.L = int(encoder_length)
        self.D = int(decoder_length)
        self.register_buffer("ts", torch.as_tensor(np.asarray(target_scale, np.float32)).view(1, -1))

    def forward(self, encoder_cont: torch.Tensor, decoder_cont: torch.Tensor) -> torch.Tensor:
        b = encoder_cont.shape[0]
        x = {
            "encoder_cont": encoder_cont,
            "decoder_cont": decoder_cont,
            "encoder_cat": torch.zeros((b, self.L, 0), dtype=torch.long),
            "decoder_cat": torch.zeros((b, self.D, 0), dtype=torch.long),
            "encoder_target": encoder_cont[..., 0] * 0.0,
            "decoder_target": decoder_cont[..., 0] * 0.0,
            "encoder_lengths": torch.full((b,), self.L, dtype=torch.long),
            "decoder_lengths": torch.full((b,), self.D, dtype=torch.long),
            "decoder_time_idx": torch.arange(self.D, dtype=torch.long).expand(b, self.D),
            "groups": torch.zeros((b, 1), dtype=torch.long),
            "target_scale": self.ts.expand(b, -1),
        }
        return self.model(x)["prediction"]


def scale(raw: np.ndarray, columns: list[str], scalers: dict) -> np.ndarray:
    c = np.array([scalers[k]["center"] for k in columns], dtype=np.float64)
    s = np.array([scalers[k]["scale"] for k in columns], dtype=np.float64)
    return ((np.asarray(raw, np.float64) - c) / s).astype(np.float32)


def nhits_tensors(seq: np.ndarray, columns: list[str], scalers: dict, L: int, D: int):
    """seq: (n, R) unscaled rows in `columns` order. Returns tensors for every end index k
    (encoder = rows k-L+1..k, left-padded with row 0 when k < L-1)."""
    sc = scale(seq, columns, scalers)
    n = len(sc)
    pad = np.concatenate([np.repeat(sc[:1], L - 1, axis=0), sc], axis=0)
    idx = np.arange(n)[:, None] + np.arange(L)[None, :]
    enc = pad[idx]                                         # (n, L, R)
    dec = np.repeat(enc[:, -1:, :], D, axis=1)             # unused by N-HiTS (no known reals)
    return enc, dec


def tft_tensors(seq: np.ndarray, columns: list[str], scalers: dict, L: int):
    """Full-length TFT tensors for every window k >= L (encoder = k-L..k-1, decoder = k)."""
    sc = scale(seq, columns, scalers)
    n = len(sc)
    if n <= L:
        return np.zeros((0, L, len(columns)), np.float32), np.zeros((0, 1, len(columns)), np.float32)
    idx = np.arange(L, n)[:, None] + np.arange(-L, 0)[None, :]
    return sc[idx], sc[np.arange(L, n)][:, None, :]


def torch_run(wrapper: nn.Module, enc: np.ndarray, dec: np.ndarray, batch: int = 4096) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(enc), batch):
            outs.append(wrapper(torch.as_tensor(enc[i:i + batch]), torch.as_tensor(dec[i:i + batch])).numpy())
    return np.concatenate(outs) if outs else np.zeros((0,))


def softmax(z: np.ndarray, axis=-1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
