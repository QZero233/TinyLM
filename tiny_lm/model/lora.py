from typing import Any, Optional, Dict, Tuple, List

import json
import os

import torch
from torch import nn
from tiny_lm.data.checkpoint import load_torch_checkpoint
from .common import Linear

class LoRAConfig:
    def __init__(self, module_name: str, b: torch.Tensor, a: torch.Tensor):
        self.module_name = module_name
        self.b = b
        self.a = a


def save_lora_configs(lora_configs: List[LoRAConfig], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    manifest = []
    for idx, config in enumerate(lora_configs):
        b_file = f"lora_{idx}_b.pt"
        a_file = f"lora_{idx}_a.pt"
        torch.save(config.b.detach().cpu(), os.path.join(save_dir, b_file))
        torch.save(config.a.detach().cpu(), os.path.join(save_dir, a_file))
        manifest.append({
            "module_name": config.module_name,
            "b_file": b_file,
            "a_file": a_file,
        })

    manifest_file = os.path.join(save_dir, "lora_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def save_lora_checkpoint(lora_configs: List[LoRAConfig], save_file: str) -> None:
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    state = {
        "lora": [
            {
                "module_name": cfg.module_name,
                "b": cfg.b.detach().cpu(),
                "a": cfg.a.detach().cpu(),
            }
            for cfg in lora_configs
        ]
    }
    torch.save(state, save_file)


def save_lora_trainable_checkpoint(model: nn.Module, step: int, save_file: str) -> None:
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    state = {
        "t": int(step),
        "trainable_model": trainable_state,
    }
    torch.save(state, save_file)


def load_lora_trainable_checkpoint(load_file: str, model: nn.Module | None = None) -> int:
    state = load_torch_checkpoint(load_file, map_location="cpu")
    step = int(state.get("t", 0))
    trainable_state = state.get("trainable_model")
    if model is not None:
        if not isinstance(trainable_state, dict):
            raise ValueError(f"Invalid lora trainable checkpoint format: {load_file}")
        model.load_state_dict(trainable_state, strict=False)
    return step


def load_lora_configs(load_dir: str) -> List[LoRAConfig]:
    if os.path.isfile(load_dir):
        state = load_torch_checkpoint(load_dir, map_location="cpu")
        if "trainable_model" in state:
            items = []
            for key, tensor in state["trainable_model"].items():
                if key.endswith(".b"):
                    module_name = key[:-2]
                    a_key = f"{module_name}.a"
                    if a_key in state["trainable_model"]:
                        items.append(
                            LoRAConfig(
                                module_name,
                                state["trainable_model"][key],
                                state["trainable_model"][a_key],
                            )
                        )
            return items
        items = state.get("lora", [])
        return [LoRAConfig(item["module_name"], item["b"], item["a"]) for item in items]

    manifest_file = os.path.join(load_dir, "lora_manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lora_configs = []
    for item in manifest:
        b = load_torch_checkpoint(os.path.join(load_dir, item["b_file"]), map_location="cpu")
        a = load_torch_checkpoint(os.path.join(load_dir, item["a_file"]), map_location="cpu")
        lora_configs.append(LoRAConfig(item["module_name"], b, a))
    return lora_configs


class LoRALinear(nn.Module):
    # Linear: (in, out)
    # B: (in, r)
    # A: (r, out)
    def __init__(self, origin_linear: Linear, b: Optional[torch.Tensor], a: Optional[torch.Tensor], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.origin_linear = origin_linear
        self.b = nn.Parameter(b)
        self.a = nn.Parameter(a)

    def forward(self, x):
        return self.origin_linear(x) + x@self.b@self.a
