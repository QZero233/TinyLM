import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from tiny_lm.config import ModelConfig


DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[2] / "configs" / "train_zh.json"


def _get_model_section(raw_config: dict[str, Any]) -> dict[str, Any]:
    model_values = raw_config.get("model", raw_config)
    if not isinstance(model_values, dict):
        raise ValueError("model config must be an object")
    return model_values


def _build_model_config(model_values: dict[str, Any]) -> ModelConfig:
    valid_fields = {field.name for field in fields(ModelConfig)}
    unknown_fields = set(model_values) - valid_fields
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown model config field(s): {unknown}")

    return ModelConfig(**model_values)


def get_config(config_file: str | Path = DEFAULT_CONFIG_FILE) -> LlamaConfig:
    with open(config_file) as f:
        raw_config = json.load(f)

    model_config = _build_model_config(_get_model_section(raw_config))
    return LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=model_config.d_ff,
        num_hidden_layers=model_config.num_layers,
        num_attention_heads=model_config.num_heads,
        num_key_value_heads=model_config.num_heads,
        max_position_embeddings=model_config.context_length,
        rope_theta=model_config.theta,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
    )


def get_model(config_file: str | Path = DEFAULT_CONFIG_FILE) -> nn.Module:
    model = LlamaForCausalLM(get_config(config_file))
    with open(config_file) as f:
        model_values = _get_model_section(json.load(f))
    if model_values.get("gradient_checkpoint", False):
        model.gradient_checkpointing_enable()

    return model
