import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import yaml


@dataclass
class FocusPlan:
    """Plan describing how frequently focus tokens are injected into sequences."""

    block_len: int
    focus_tokens_per_block: int
    insert_every_n: int
    condense_ratio: int = 64

    def blocks_for_length(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.insert_every_n)

    @property
    def focus_interval(self) -> int:
        return self.insert_every_n


def load_yaml(path: str) -> Dict:
    def _expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(_expand(v) for v in value)
        return value

    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return _expand(loaded)


def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def ensure_token(tokenizer, token: str) -> int:
    if token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    return tokenizer.convert_tokens_to_ids(token)


def set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(trainable)


@contextlib.contextmanager
def timing(description: str, sink: Optional[List[str]] = None):
    start = time.time()
    yield
    dur = time.time() - start
    if sink is not None:
        sink.append(f"{description}:{dur:.4f}")


def chunk_tensor(tensor: torch.Tensor, chunk_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, tensor.size(-2), chunk_size):
        yield tensor[..., start : start + chunk_size, :]


def rolling_window(sequence: Sequence[int], window: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, len(sequence), window):
        yield start, min(start + window, len(sequence))


def describe_memory(device: torch.device) -> Dict[str, float]:
    if not device.type.startswith("cuda"):
        return {"allocated": 0.0, "reserved": 0.0}
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(device)
    return {"allocated_mb": allocated, "reserved_mb": reserved}


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class NullWriter:
    def write(self, *_: Iterable) -> None:  # pragma: no cover - debugging convenience
        pass

    def flush(self) -> None:
        pass
