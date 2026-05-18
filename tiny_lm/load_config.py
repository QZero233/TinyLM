import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, TypeVar
from typing import Tuple

from transformers import AutoTokenizer

from .config import ModelConfig, OptimizerConfig
from .data.zh_dataset import TinyLMZhDataset


DEFAULT_TRAIN_CONFIG = "/root/autodl-tmp/TinyLM/configs/train_zh.json"


@dataclass
class TrainConfig:
    project_name: str
    tokenizer_path: str
    checkpoint: str
    data_dir: str
    zh_token_dtype: str
    checkpoint_base_dir: str
    epochs: int
    gradient_accumulate: int
    checkpoint_save_accum_steps: int = 200
    valid_steps: int = 200
    fix_lr: float | None = None
    print_optimizer_update_ratio: bool = False


@dataclass
class LoraTrainConfig:
    r: int
    checkpoint_base_dir: str
    data_dir: str
    base_model_checkpoint: str
    lora_checkpoint: str = ""
    full_finetune: bool = False
    mask_question: bool = True
    fix_lr: float | None = None
    print_optimizer_update_ratio: bool = False
    auto_resize_embedding: bool = True
    valid_steps: int = 200
    checkpoint_save_steps: int = 200
    lr_scheduler_warmup_steps: int = 3600
    lr_scheduler_total_steps: int = 120000
    lr_scheduler_min_lr: float = 3e-6
    lr_scheduler_max_lr: float = 1e-5


def get_tokenizer(tokenizer_path: str) -> Any:
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def load_tokenizer(tokenizer_path: str) -> Any:
    return get_tokenizer(tokenizer_path)


def get_dataset(data_dir: str, seq_len: int, zh_token_dtype: str = "uint16",
                full_random: bool = True, train: bool = True) -> TinyLMZhDataset:
    return TinyLMZhDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        dtype=zh_token_dtype,
        full_random=full_random,
        train=train,
    )


DataclassT = TypeVar("DataclassT")


def _build_dataclass(config_cls: type[DataclassT], values: dict[str, Any], section_name: str,
                     extra_values: dict[str, Any] | None = None) -> DataclassT:
    dataclass_fields = fields(config_cls)
    valid_fields = {field.name for field in dataclass_fields}
    extra_values = extra_values or {}
    required_fields = {
        field.name for field in dataclass_fields
        if field.name not in extra_values
        and field.default is MISSING
        and field.default_factory is MISSING
    }
    missing_fields = required_fields - set(values)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing required config field(s) in '{section_name}': {missing}")

    for key, value in values.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown config field '{section_name}.{key}' for {config_cls.__name__}")

    return config_cls(**extra_values, **values)


def _get_required_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    if section_name not in raw_config:
        raise ValueError(f"Missing required config section '{section_name}'")
    section = raw_config[section_name]
    if not isinstance(section, dict):
        raise ValueError(f"'{section_name}' config must be an object")
    return section


def load_train_config(config_file: str) -> Tuple[TrainConfig, ModelConfig, OptimizerConfig]:
    with open(config_file, "r") as f:
        raw_config = json.load(f)

    training_values = _get_required_section(raw_config, "training")
    model_values = _get_required_section(raw_config, "model")
    optimizer_values = _get_required_section(raw_config, "optimizer")
    training_core_values = dict(training_values)
    training_core_values.pop("lora", None)
    train_config = _build_dataclass(TrainConfig, training_core_values, "training")

    if train_config.zh_token_dtype not in ("uint16", "uint32", "int32", "int64"):
        raise ValueError("training.zh_token_dtype must be one of: uint16, uint32, int32, int64")

    model_config = _build_dataclass(ModelConfig, model_values, "model")
    optimizer_config = _build_dataclass(OptimizerConfig, optimizer_values, "optimizer")

    return train_config, model_config, optimizer_config


def load_lora_train_config(config_file: str) -> Tuple[TrainConfig, ModelConfig, OptimizerConfig, LoraTrainConfig]:
    train_config, model_config, optimizer_config = load_train_config(config_file)
    with open(config_file, "r") as f:
        raw_config = json.load(f)

    training_values = _get_required_section(raw_config, "training")
    lora_values = _get_required_section(training_values, "lora")
    lora_config = _build_dataclass(LoraTrainConfig, lora_values, "training.lora")
    if lora_config.r <= 0:
        raise ValueError("training.lora.r must be > 0")
    if not lora_config.base_model_checkpoint:
        raise ValueError("training.lora.base_model_checkpoint must not be empty")
    if lora_config.fix_lr is not None and lora_config.fix_lr <= 0:
        raise ValueError("training.lora.fix_lr must be > 0 when set")
    return train_config, model_config, optimizer_config, lora_config
