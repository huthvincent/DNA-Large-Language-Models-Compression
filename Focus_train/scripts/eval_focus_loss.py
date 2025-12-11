#!/usr/bin/env python3
"""
Compute average next-token loss for Focus-Token fine-tuned runs.

Example:
    python scripts/eval_focus_loss.py \
        --run-dir output/focus_runs/focus_chr1 \
        --chroms 2 \
        --segment-length 60000 \
        --num-seqs 1000
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from focus.adapter import LLMAdapter
from focus.token_injection import FocusTokenInserter
from focus.train import compute_loss
from focus.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRCH38_DIR = DATA_DIR / "GRCh38"


@dataclass
class TextSample:
    text: str
    chrom: str
    index: int


class TokenDataset(Dataset):
    def __init__(self, inputs: List[Dict[str, torch.Tensor]]) -> None:
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.inputs[idx]


def normalize_chrom(token: str) -> str:
    token = token.strip()
    if not token.lower().startswith("chr"):
        token = f"chr{token}"
    return token.replace("CHR", "chr")


def read_fasta_sequences(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith(">"):
                continue
            seq = line.strip().upper()
            if seq:
                yield seq


def iter_segments(
    chrom: str,
    segment_length: int,
) -> Iterator[str]:
    fasta_path = GRCH38_DIR / f"{chrom}.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"Missing FASTA for {chrom}: {fasta_path}")

    buffer: List[str] = []
    buffered = 0

    def flush(force: bool = False) -> Iterator[str]:
        nonlocal buffer, buffered
        if not buffer:
            return iter(())
        joined = "".join(buffer)
        total = len(joined)
        produced: List[str] = []
        idx = 0
        while total - idx >= segment_length:
            produced.append(joined[idx : idx + segment_length])
            idx += segment_length
        remainder = joined[idx:]
        if force and remainder:
            produced.append(remainder)
            remainder = ""
        buffer = [remainder] if remainder else []
        buffered = len(remainder)
        return iter(produced)

    for seq in read_fasta_sequences(fasta_path):
        buffer.append(seq)
        buffered += len(seq)
        if buffered >= segment_length:
            for segment in flush():
                yield segment

    for segment in flush(force=True):
        yield segment


def build_samples(
    chroms: Sequence[str],
    segment_length: int,
    num_seqs: Optional[int],
) -> List[TextSample]:
    samples: List[TextSample] = []
    for chrom_raw in chroms:
        chrom = normalize_chrom(chrom_raw)
        count = 0
        for idx, segment in enumerate(iter_segments(chrom, segment_length)):
            samples.append(TextSample(text=segment, chrom=chrom, index=idx))
            count += 1
            if num_seqs is not None and count >= num_seqs:
                break
        if count == 0:
            raise ValueError(f"No segments collected for {chrom}.")
    return samples


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate average loss on selected chromosomes.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Focus Token run directory (e.g. output/focus_runs/focus_chr1)")
    parser.add_argument("--chroms", type=str, required=True, help="Comma-separated chromosome list, e.g. '2' or '1,2,X'")
    parser.add_argument("--segment-length", type=int, default=60_000, help="Sequence length (in bases) per segment before tokenization.")
    parser.add_argument("--num-seqs", type=int, default=None, help="Limit number of segments per chromosome (None = use all).")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max token length (defaults to block_len in config).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    chroms = [normalize_chrom(c) for c in args.chroms.split(",") if c.strip()]
    if not chroms:
        raise ValueError("Chromosome list is empty.")

    samples = build_samples(chroms, args.segment_length, args.num_seqs)
    print(f"Collected {len(samples)} segments from {', '.join(chroms)}.")

    config_path = args.run_dir / "configs" / "focus_config.generated.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_yaml(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = LLMAdapter.from_pretrained(cfg["model_name_or_path"], cfg, device=device)

    checkpoint_path = args.run_dir / "checkpoints" / "focus_adapter.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    adapter.focus_attention.load_state_dict(state["focus_attention"])
    adapter.model.get_input_embeddings().load_state_dict(state["embedding"])
    adapter.eval()

    inserter = FocusTokenInserter(cfg, adapter.tokenizer)

    block_len = args.max_tokens or int(cfg.get("block_len", 1024))
    tokenizer = adapter.tokenizer

    tokenized_inputs: List[Dict[str, torch.Tensor]] = []
    for sample in samples:
        encoded = tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=block_len,
            return_tensors="pt",
        )
        tokenized_inputs.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )

    dataset = TokenDataset(tokenized_inputs)
    def move_to_device(batch):
        collated = default_collate(batch)
        return {k: v.to(device) for k, v in collated.items()}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=move_to_device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            updated, meta = inserter.prepare_training_batch(batch)
            outputs = adapter.forward_with_focus_tokens(
                updated["input_ids"],
                updated["attention_mask"],
                meta.focus_mask,
                causal_mask=meta.causal_mask,
                block_map=meta.block_map,
                use_cache=False,
            )
            loss = compute_loss(outputs.logits, updated["input_ids"], adapter.tokenizer.pad_token_id)
            tokens = updated["attention_mask"].sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    if total_tokens == 0:
        print("No valid tokens encountered.")
        return 1

    avg_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Average loss: {avg_loss:.6f} (per-token CE)")
    print(f"Perplexity: {perplexity:.6f}")
    print(f"Total tokens evaluated: {total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
