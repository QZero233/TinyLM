from typing import Optional, Tuple

import torch

from .config import ModelConfig, OptimizerConfig
from .data import load_checkpoint
from .model import Transformer
from .optmize import AdamW


def init_model(config: ModelConfig) -> torch.nn.Module:
    return Transformer(
        config.vocab_size,
        config.context_length,
        config.num_layers,
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.theta,
        gradient_checkpoint=config.gradient_checkpoint,
    )


def init_optimizer(config: OptimizerConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        config.lr,
        (config.beta_1, config.beta_2),
        config.weight_decay,
        config.eps,
    )


def load_model_checkpoint(checkpoint: Optional[str], model: torch.nn.Module) -> int:
    if checkpoint is None:
        return 0
    return load_checkpoint(checkpoint, model, None)


def load_optimizer_checkpoint(checkpoint: Optional[str], optimizer: torch.optim.Optimizer) -> int:
    if checkpoint is None:
        return 0
    return load_checkpoint(checkpoint, None, optimizer)


def init_model_from_checkpoint(config: ModelConfig, checkpoint: Optional[str]) -> torch.nn.Module:
    model = init_model(config)
    load_model_checkpoint(checkpoint, model)
    return model


def init_optimizer_from_checkpoint(
    config: OptimizerConfig,
    model: torch.nn.Module,
    checkpoint: Optional[str],
) -> Tuple[torch.optim.Optimizer, int]:
    optimizer = init_optimizer(config, model)
    t = load_optimizer_checkpoint(checkpoint, optimizer)
    return optimizer, t
