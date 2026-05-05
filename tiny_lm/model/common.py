import math
from typing import Any, Optional

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.zeros((out_features, in_features)))

        init_std = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weights, mean=0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

class SiLU(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_ff = d_ff

        if d_ff is None:
            d_ff = (8 * d_model // 3)
            self.d_ff = (d_ff + 63) // 64 * 64

        self.W1 = nn.Parameter(torch.empty((self.d_ff, self.d_model)))
        self.W2 = nn.Parameter(torch.empty((self.d_model, self.d_ff)))
        self.W3 = nn.Parameter(torch.empty((self.d_ff, self.d_model)))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp1 = x @ self.W1.T
        tmp1 = self.silu(tmp1)

        tmp2 = x @ self.W3.T

        tmp3 = tmp1 * tmp2
        return tmp3 @ self.W2.T

class Softmax(nn.Module):
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_val, _ = torch.max(x, dim=dim, keepdim=True)
        x -= max_val
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        return x_exp / x_exp_sum

