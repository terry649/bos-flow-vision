"""RAFT (Teed and Deng 2020) via torchvision, used zero-shot with pretrained weights.

API per torchvision 0.29: ``raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)``;
inputs are (B, 3, H, W) in [-1, 1] with H and W divisible by 8; forward returns
the list of iterative flow estimates, the last being the final one, with channel
0 the x (column) displacement.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=4)
def _model(variant: str, device: str):
    if variant == "large":
        m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
    elif variant == "small":
        m = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False)
    else:
        raise ValueError(variant)
    return m.eval().to(device)


def _to_tensor(img: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    t = t[None, None].expand(1, 3, *img.shape)  # grayscale to 3 channels
    return (t * 2.0 - 1.0).to(device)


@torch.inference_mode()
def raft(ref: np.ndarray, defl: np.ndarray, variant: str = "large",
         num_flow_updates: int = 12, device: torch.device | None = None) -> np.ndarray:
    device = device or default_device()
    model = _model(variant, str(device))
    rows, cols = ref.shape
    pr, pc = (-rows) % 8, (-cols) % 8
    a, b = _to_tensor(ref, device), _to_tensor(defl, device)
    if pr or pc:
        a = F.pad(a, (0, pc, 0, pr), mode="replicate")
        b = F.pad(b, (0, pc, 0, pr), mode="replicate")
    flow = model(a, b, num_flow_updates=num_flow_updates)[-1][0, :, :rows, :cols]
    return flow.float().cpu().numpy()
