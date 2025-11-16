#!/usr/bin/env python3
"""
Measure GPU memory consumption of Evo-2 7B (base) vs the summary-token (FOCUS) adapter.

Random DNA sequences of varying lengths are generated, streamed through both models
using chunked inference (default chunk/block length = 1024). The script records peak
GPU memory usage per length and writes the results to CSV.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from beacon.adapter import LLMAdapter
from beacon.token_injection import BeaconInserter
from beacon.utils import load_yaml


DEFAULT_LENGTHS = [
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    30_000,
    40_000,
    50_000,
    60_000,
    700_000,
]


def parse_lengths(raw: Optional[str]) -> List[int]:
    if not raw:
        return DEFAULT_LENGTHS
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPU memory usage across sequence lengths.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Summary-adapter run directory (focus_runs/chr1).")
    parser.add_argument("--base-model", type=Path, required=True, help="Path to the original Evo-2 7B checkpoint.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Tokens processed per forward pass when streaming. Defaults to block_len from beacon config.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Comma-separated list of sequence lengths (defaults to preset list).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sequence generation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to run_dir/outputs/memory/memory.csv).",
    )
    parser.add_argument(
        "--save-fasta",
        action="store_true",
        help="Also save generated sequences as FASTA under outputs/memory/random_sequences.fasta",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def generate_sequences(lengths: Sequence[int], seed: int) -> Dict[int, str]:
    rng = random.Random(seed)
    alphabet = "ACGT"
    sequences: Dict[int, str] = {}
    for length in lengths:
        seq = "".join(rng.choice(alphabet) for _ in range(length))
        sequences[length] = seq
    return sequences


def save_fasta(path: Path, sequences: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for length, seq in sequences.items():
            fh.write(f">seq_len_{length}\n")
            for start in range(0, len(seq), 120):
                fh.write(seq[start : start + 120] + "\n")


def tokenize_sequence(tokenizer, sequence: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    return input_ids


def profile_cuda(device: torch.device, fn) -> Dict[str, float]:
    metrics = {
        "peak_alloc_mb": float("nan"),
        "peak_reserved_mb": float("nan"),
        "status": "ok",
        "message": "",
    }
    if device.type != "cuda":
        fn()
        return metrics

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        fn()
        torch.cuda.synchronize(device)
        metrics["peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            metrics["status"] = "oom"
            metrics["message"] = str(exc)
        else:
            raise
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return metrics


def forward_full_sequence(adapter: LLMAdapter, token_ids: torch.Tensor) -> Dict[str, int]:
    total_len = token_ids.size(1)
    if total_len == 0:
        return {"beacons": 0, "sequence_len": 0}
    attention_mask = torch.ones_like(token_ids, dtype=torch.long)
    with torch.no_grad():
        adapter.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    return {"beacons": 0, "sequence_len": int(attention_mask.sum().item())}


def stream_base_model(adapter: LLMAdapter, token_ids: torch.Tensor, chunk_size: int) -> Dict[str, int]:
    total_len = token_ids.size(1)
    if total_len == 0:
        return {"beacons": 0, "sequence_len": 0}

    device = token_ids.device
    offset = 0
    past = None
    attention_mask = torch.zeros((1, 0), dtype=torch.long, device=device)

    with torch.no_grad():
        while offset < total_len:
            chunk = token_ids[:, offset : offset + chunk_size]
            chunk_len = chunk.size(1)
            if chunk_len == 0:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, chunk_len), dtype=torch.long, device=device)],
                dim=1,
            )
            outputs = adapter.model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            offset += chunk_len

    return {"beacons": 0, "sequence_len": int(attention_mask.size(1))}


def stream_beacon_model(
    adapter: LLMAdapter,
    inserter: BeaconInserter,
    token_ids: torch.Tensor,
    chunk_size: int,
) -> Dict[str, int]:
    total_len = token_ids.size(1)
    if total_len == 0:
        return {"beacons": 0, "sequence_len": 0}

    device = token_ids.device
    state = inserter.init_inference_state()
    attention_mask = torch.zeros((1, 0), dtype=torch.long, device=device)
    beacon_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
    past = None
    offset = 0

    with torch.no_grad():
        while offset < total_len:
            chunk = token_ids[:, offset : offset + chunk_size]
            chunk_len = chunk.size(1)
            if chunk_len == 0:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, chunk_len), dtype=torch.long, device=device)],
                dim=1,
            )
            beacon_mask = torch.cat(
                [beacon_mask, torch.zeros((1, chunk_len), dtype=torch.bool, device=device)],
                dim=1,
            )
            outputs = adapter.model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            offset += chunk_len

            for _ in range(chunk_len):
                inserter.observe_token(state)
                insertion = inserter.maybe_insert_beacons(state)
                if insertion:
                    beacon_tokens = torch.tensor(
                        insertion.beacon_ids,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(beacon_tokens, dtype=torch.long)],
                        dim=1,
                    )
                    beacon_flags = torch.ones_like(beacon_tokens, dtype=torch.bool)
                    beacon_mask = torch.cat([beacon_mask, beacon_flags], dim=1)
                    outputs = adapter.model(
                        input_ids=beacon_tokens,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = adapter.replace_or_merge_kv(outputs.past_key_values, beacon_mask[0])

    return {
        "beacons": int(beacon_mask.sum().item()),
        "sequence_len": int(attention_mask.size(1)),
    }


def evaluate_model(
    label: str,
    adapter: LLMAdapter,
    sequences: Dict[int, str],
    device: torch.device,
    inserter: Optional[BeaconInserter] = None,
    chunk_size: Optional[int] = None,
    stream: bool = False,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    tokenizer = adapter.tokenizer
    for length in sorted(sequences):
        sequence = sequences[length]
        token_ids = tokenize_sequence(tokenizer, sequence, device)
        token_count = int(token_ids.size(1))

        extra_info = {"beacons": 0, "sequence_len": token_count}
        print(f"[INFO] {label}: length={length} chars ({token_count} tokens)", flush=True)

        def _run():
            nonlocal extra_info
            if not stream or inserter is None:
                extra_info = forward_full_sequence(adapter, token_ids)
            else:
                if chunk_size is None:
                    raise ValueError("chunk_size must be provided when streaming.")
                extra_info = stream_beacon_model(adapter, inserter, token_ids, chunk_size)

        metrics = profile_cuda(device, _run)
        row: Dict[str, object] = {
            "model": label,
            "sequence_chars": length,
            "tokens": token_count,
            "peak_alloc_mb": metrics["peak_alloc_mb"],
            "peak_reserved_mb": metrics["peak_reserved_mb"],
            "status": metrics["status"],
            "message": metrics["message"],
        }
        row["beacons"] = extra_info["beacons"]
        row["effective_sequence_len"] = extra_info["sequence_len"]
        results.append(row)
        del token_ids
    return results


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device required for memory evaluation.")

    cfg_path = args.run_dir / "configs" / "beacon_config.generated.yaml"
    cfg = load_yaml(str(cfg_path))
    lengths = parse_lengths(args.lengths)
    sequences = generate_sequences(lengths, args.seed)

    output_path = args.output or (args.run_dir / "outputs" / "memory" / "memory.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_fasta:
        fasta_path = output_path.parent / "random_sequences.fasta"
        save_fasta(fasta_path, sequences)

    base_cfg = dict(cfg)
    base_cfg["model_name_or_path"] = str(args.base_model)

    print(f"[INFO] Loading base model from {base_cfg['model_name_or_path']}")
    base_adapter = LLMAdapter.from_pretrained(base_cfg["model_name_or_path"], base_cfg, device=device)
    base_adapter.eval()

    chunk_size = args.chunk_size or int(cfg.get("block_len", 1024))

    base_results = evaluate_model("base_evo2", base_adapter, sequences, device, stream=False)
    del base_adapter
    torch.cuda.empty_cache()

    print(f"[INFO] Loading summary adapter from {cfg.get('model_name_or_path')}")
    ab_adapter = LLMAdapter.from_pretrained(cfg["model_name_or_path"], cfg, device=device)
    checkpoint = args.run_dir / "checkpoints" / "beacon_adapter.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        ab_adapter.beacon_attention.load_state_dict(state["beacon_attention"])
        ab_adapter.model.get_input_embeddings().load_state_dict(state["embedding"])
    else:
        raise FileNotFoundError(f"Beacon checkpoint not found: {checkpoint}")
    ab_adapter.eval()
    inserter = BeaconInserter(cfg, ab_adapter.tokenizer)

    ab_results = evaluate_model(
        "activation_beacon",
        ab_adapter,
        sequences,
        device,
        inserter=inserter,
        chunk_size=chunk_size,
        stream=True,
    )
    del ab_adapter
    torch.cuda.empty_cache()

    fieldnames = [
        "model",
        "sequence_chars",
        "tokens",
        "beacons",
        "effective_sequence_len",
        "peak_alloc_mb",
        "peak_reserved_mb",
        "status",
        "message",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_results + ab_results:
            writer.writerow(row)

    print(f"[INFO] Memory metrics saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
