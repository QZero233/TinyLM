import math
from typing import Any, Callable, Tuple

import torch
from torch import nn
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float, betas: Tuple[float, float], weight_decay: float, eps: float = 1e-8):
        beta_1, beta_2 = betas
        defaults = {
            "lr": lr,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            decay = group["decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))

                lr_adjusted = lr * (math.sqrt(1 - beta_2**t)) / (1 - beta_1**t)
                p.data *= 1 - lr * decay # Weight decay

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2

                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss

def cosine_lr_scheduler(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return t * lr_max / t_w
    elif t_w <= t <= t_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clip(params: ParamsT, m: float, eps: float = 1e-6):
    norm_sum = 0
    for param in params:
        if param.grad is None:
            continue
        norm = torch.norm(param.grad.data, p=2)
        norm_sum = norm**2 + norm_sum
    norm_sum = torch.sqrt(norm_sum)

    if norm_sum > m:
        factor = m / (norm_sum + eps)
        for param in params:
            if param.grad is not None:
                param.grad.data *= factor

