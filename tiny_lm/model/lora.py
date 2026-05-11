from typing import Any, Optional, Dict, Tuple, List

import json
import os

import torch
from torch import nn
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


def load_lora_configs(load_dir: str) -> List[LoRAConfig]:
    manifest_file = os.path.join(load_dir, "lora_manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lora_configs = []
    for item in manifest:
        b = torch.load(os.path.join(load_dir, item["b_file"]), map_location="cpu")
        a = torch.load(os.path.join(load_dir, item["a_file"]), map_location="cpu")
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
