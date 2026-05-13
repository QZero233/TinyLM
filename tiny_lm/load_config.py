import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, TypeVar
from typing import Tuple

from transformers import AutoTokenizer
from transformers import SentencePieceBackend, TokenizersBackend

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


def get_tokenizer(tokenizer_path: str) -> TokenizersBackend | SentencePieceBackend:
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def load_tokenizer(tokenizer_path: str) -> TokenizersBackend | SentencePieceBackend:
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


def get_tokenizer_vocab_size(tokenizer_path: str) -> int:
    tokenizer = get_tokenizer(tokenizer_path)
    return len(tokenizer)


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
    train_config = _build_dataclass(TrainConfig, training_values, "training")

    if train_config.zh_token_dtype not in ("uint16", "uint32", "int32", "int64"):
        raise ValueError("training.zh_token_dtype must be one of: uint16, uint32, int32, int64")

    vocab_size = get_tokenizer_vocab_size(train_config.tokenizer_path)
    model_config = _build_dataclass(ModelConfig, model_values, "model", extra_values={"vocab_size": vocab_size})
    optimizer_config = _build_dataclass(OptimizerConfig, optimizer_values, "optimizer")

    return train_config, model_config, optimizer_config
