import json
import os
import sys
from dataclasses import MISSING, dataclass, fields
from typing import Tuple

from .config import ModelConfig, OptimizerConfig
from .data.dataloader import TinyStoryDataset
from .data.zh_dataset import TinyLMZhDataset
from .tokenizer import BPETokenizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAIN_CONFIG = os.path.join(PROJECT_ROOT, "configs", "train_zh.json")

TOKENIZER_REGISTRY = {
    "gpt": {
        "type": "bpe",
        "path": "/root/autodl-tmp/CS336_A1/saved_gpt_tokenizer",
    },
    "gpt_inst": {
        "type": "bpe",
        "path": "/root/autodl-tmp/CS336_A1/saved_gpt_inst_tokenizer",
    },
    "chatglm3": {
        "type": "chatglm",
        "path": "/root/autodl-tmp/zh_data/chatglm3_tokenizer",
    },
}

DATASET_REGISTRY = {
    "tinystory": {
        "factory": TinyStoryDataset,
        "kwargs": lambda data_dir, seq_len, zh_token_dtype, zh_fold, train: {
            "data_dir": data_dir,
            "seq_len": seq_len,
            "train": train,
        },
    },
    "zh": {
        "factory": TinyLMZhDataset,
        "kwargs": lambda data_dir, seq_len, zh_token_dtype, zh_fold, train: {
            "data_dir": data_dir,
            "seq_len": seq_len,
            "dtype": zh_token_dtype,
            "fold": zh_fold,
        },
    },
}


@dataclass
class TrainConfig:
    tokenizer_name: str
    checkpoint: str
    data_dir: str
    dataset_type: str
    zh_token_dtype: str
    checkpoint_base_dir: str
    epochs: int
    gradient_accumulate: int
    zh_fold: int | None = None


def get_tokenizer(tokenizer_name: str):
    tokenizer_info = _get_tokenizer_info(tokenizer_name)
    if tokenizer_info["type"] == "bpe":
        return BPETokenizer.from_files(tokenizer_info["path"])
    if tokenizer_info["type"] == "chatglm":
        return _load_chatglm_tokenizer(tokenizer_info["path"])
    raise ValueError(f"Unsupported tokenizer type '{tokenizer_info['type']}' for {tokenizer_name}")


def load_tokenizer(tokenizer_dir: str) -> BPETokenizer:
    return BPETokenizer.from_files(tokenizer_dir)


def get_dataset(dataset_name: str, data_dir: str, seq_len: int, zh_token_dtype: str = "uint16",
                zh_fold: int | None = None, train: bool = True):
    dataset_info = _get_dataset_info(dataset_name)
    dataset_factory = dataset_info["factory"]
    dataset_kwargs = dataset_info["kwargs"](data_dir, seq_len, zh_token_dtype, zh_fold, train)
    return dataset_factory(**dataset_kwargs)


def _get_tokenizer_info(tokenizer_name: str) -> dict:
    if tokenizer_name not in TOKENIZER_REGISTRY:
        valid_names = ", ".join(sorted(TOKENIZER_REGISTRY))
        raise ValueError(f"Unknown tokenizer_name '{tokenizer_name}'. Available tokenizers: {valid_names}")
    return TOKENIZER_REGISTRY[tokenizer_name]


def _get_dataset_info(dataset_name: str) -> dict:
    if dataset_name not in DATASET_REGISTRY:
        valid_names = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset_type '{dataset_name}'. Available datasets: {valid_names}")
    return DATASET_REGISTRY[dataset_name]


def _get_chatglm_vocab_size(tokenizer_dir: str) -> int:
    tokenizer_model = os.path.join(tokenizer_dir, "tokenizer.model")
    if os.path.exists(tokenizer_model):
        sys.path.insert(0, tokenizer_dir)
        try:
            from tokenization_chatglm import SPTokenizer
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "ChatGLM tokenizer requires transformers and sentencepiece. "
                "Run with an environment that has them installed, e.g. "
                "/root/autodl-tmp/qwen3.6/.venv/bin/python tiny_lm/train_model.py --config ..."
            ) from e
        return SPTokenizer(tokenizer_model).n_words

    vocab_file = os.path.join(tokenizer_dir, "vocab.txt")
    if os.path.exists(vocab_file):
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        return max(vocab.values()) + 1

    raise FileNotFoundError(f"Cannot find ChatGLM tokenizer files in {tokenizer_dir}")


def _load_chatglm_tokenizer(tokenizer_dir: str):
    tokenizer_model = os.path.join(tokenizer_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"Cannot find ChatGLM tokenizer model in {tokenizer_dir}")

    sys.path.insert(0, tokenizer_dir)
    try:
        from tokenization_chatglm import SPTokenizer
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ChatGLM tokenizer requires transformers and sentencepiece. "
            "Run with an environment that has them installed, e.g. "
            "/root/autodl-tmp/qwen3.6/.venv/bin/python tiny_lm/eval_model.py --config ..."
        ) from e

    return SPTokenizer(tokenizer_model)


def get_tokenizer_vocab_size(tokenizer_name: str) -> int:
    tokenizer_info = _get_tokenizer_info(tokenizer_name)
    if tokenizer_info["type"] == "bpe":
        return len(BPETokenizer.from_files(tokenizer_info["path"]).vocab)
    if tokenizer_info["type"] == "chatglm":
        return _get_chatglm_vocab_size(tokenizer_info["path"])
    raise ValueError(f"Unsupported tokenizer type '{tokenizer_info['type']}' for {tokenizer_name}")


def _build_dataclass(config_cls, values: dict, section_name: str, extra_values: dict | None = None):
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


def _get_required_section(raw_config: dict, section_name: str) -> dict:
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

    _get_dataset_info(train_config.dataset_type)
    if train_config.zh_token_dtype not in ("uint16", "uint32", "int32", "int64"):
        raise ValueError("training.zh_token_dtype must be one of: uint16, uint32, int32, int64")

    vocab_size = get_tokenizer_vocab_size(train_config.tokenizer_name)
    model_config = _build_dataclass(ModelConfig, model_values, "model", extra_values={"vocab_size": vocab_size})
    optimizer_config = _build_dataclass(OptimizerConfig, optimizer_values, "optimizer")

    return train_config, model_config, optimizer_config
