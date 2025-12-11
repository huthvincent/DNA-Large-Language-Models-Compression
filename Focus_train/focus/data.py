from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset, DatasetDict, load_dataset

LOGGER = logging.getLogger(__name__)


def _read_fasta(path: Path) -> List[str]:
    sequences: List[str] = []
    buffer: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buffer:
                    sequences.append("".join(buffer))
                    buffer.clear()
            else:
                buffer.append(line)
        if buffer:
            sequences.append("".join(buffer))
    return sequences


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_jsonl(path: Path, text_field: str) -> List[str]:
    sequences: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} of {path}: {exc}") from exc
            if text_field not in record:
                raise KeyError(f"Missing '{text_field}' in record {idx} from {path}")
            sequences.append(str(record[text_field]))
    return sequences


def _split_sequences(
    sequences: List[str],
    validation_fraction: float,
    test_fraction: float,
) -> Dict[str, List[str]]:
    total = len(sequences)
    if total == 0:
        return {"train": [], "validation": [], "test": []}

    validation_fraction = min(max(validation_fraction, 0.0), 0.9)
    test_fraction = min(max(test_fraction, 0.0), 0.9)
    remaining_fraction = max(0.0, 1.0 - validation_fraction - test_fraction)

    train_count = max(1, int(total * remaining_fraction))
    validation_count = int(total * validation_fraction)
    test_count = int(total * test_fraction)

    if train_count + validation_count + test_count < total:
        train_count = total - (validation_count + test_count)

    train = sequences[:train_count]
    remainder = sequences[train_count:]
    validation = remainder[:validation_count] if remainder else train
    test = remainder[validation_count:validation_count + test_count] if remainder else train

    if not validation:
        validation = train
    if not test:
        test = train

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def _dataset_from_sequences(sequences: Dict[str, List[str]], text_field: str) -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_dict({text_field: seqs})
            for split, seqs in sequences.items()
        }
    )


def load_text_dataset_from_config(cfg: Dict) -> DatasetDict:
    if "path" not in cfg:
        dataset_name = cfg.get("name")
        dataset_config = cfg.get("config")
        LOGGER.info("Loading HuggingFace dataset %s:%s", dataset_name, dataset_config)
        return load_dataset(dataset_name, dataset_config)

    expanded = os.path.expandvars(cfg["path"])
    path = Path(expanded).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    loader = cfg.get("loader", "text").lower()
    text_field = cfg.get("text_field", "text")
    validation_fraction = float(cfg.get("validation_fraction", 0.0))
    test_fraction = float(cfg.get("test_fraction", 0.0))

    if loader in {"fasta", "fa"}:
        sequences = _read_fasta(path)
    elif loader in {"text", "txt"}:
        sequences = _read_lines(path)
    elif loader in {"jsonl", "json"}:
        sequences = _read_jsonl(path, text_field)
    else:
        raise ValueError(f"Unsupported dataset loader '{loader}' for local path '{path}'")

    if not sequences:
        LOGGER.warning("Loaded zero sequences from %s", path)

    split_sequences = _split_sequences(sequences, validation_fraction, test_fraction)
    return _dataset_from_sequences(split_sequences, text_field)


def get_dataset_split(dataset, split: str):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available}")
        return dataset[split]
    return dataset
