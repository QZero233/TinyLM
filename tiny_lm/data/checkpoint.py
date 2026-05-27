import typing
import os

import torch


def load_torch_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    map_location: str | torch.device | typing.Callable | None = "cpu",
):
    # PyTorch 2.6 changed torch.load's default to weights_only=True. Local
    # TinyLM checkpoints can include NPU tensor rebuild metadata, so load the
    # trusted project checkpoint format explicitly.
    return torch.load(src, map_location=map_location, weights_only=False)


def save_checkpoint(model: torch.nn.Module, optimizer: typing.Optional[torch.optim.Optimizer],
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    opt_state = optimizer.state_dict() if optimizer is not None else None
    state = {
        "model": model_state,
        "optimizer": opt_state,
        "t": iteration
    }

    torch.save(state, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: typing.Optional[torch.nn.Module], optimizer: typing.Optional[torch.optim.Optimizer]) -> int:
    # Always load checkpoint tensors onto CPU first. This avoids unexpected
    # CUDA allocations when a checkpoint was saved from GPU training.
    state = load_torch_checkpoint(src, map_location="cpu")

    # 处理Compiled的模型参数前缀
    if model is not None:
        model_state = state["model"]
        new_state = {}
        for k, v in model_state.items():
            new_state[k.replace("_orig_mod.", "").replace("module.", "")] = v
        state["model"] = new_state

        model.load_state_dict(state["model"], strict=True)

    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["t"]
