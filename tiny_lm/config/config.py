from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1
    context_length: int = 1024
    num_layers: int = 8
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1024
    theta: float = 10000
    gradient_checkpoint: bool = True
    batch_size: int = 16
    pytorch_impl: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-6
    reset: bool = False


def get_default_model_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size)


def get_default_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig()
