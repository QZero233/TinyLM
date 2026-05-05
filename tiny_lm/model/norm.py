from typing import Any

import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_factor = torch.sqrt((x**2).sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        result = x / rms_factor * self.gain

        return result.to(in_dtype)