#!/usr/bin/env python3
"""
Evaluate next-token metrics on 10k-length segments formed by concatenating 10 sequential 1k chunks.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from focus.adapter import LLMAdapter
from focus.token_injection import FocusTokenInserter
from focus.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
GRCH38_DIR = ROOT / "data" / "GRCh38"


@dataclass
class Segment:
    chrom: str
    start: int
    end: int
    text: str


class SegmentDataset(Dataset):
    def __init__(self, segments: Sequence[Segment], tokenizer) -> None:
        self.segments = segments
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[idx]
        encoded = self.tokenizer(
            seg.text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chrom": seg.chrom,
            "start": seg.start,
            "end": seg.end,
        }


def normalize_chrom(raw: str) -> str:
    raw = raw.strip()
    return raw if raw.lower().startswith("chr") else f"chr{raw}".replace("CHR", "chr")


def read_fasta(chrom: str) -> str:
    path = GRCH38_DIR / f"{chrom}.fasta"
    if not path.exists():
        raise FileNotFoundError(f"FASTA for {chrom} not found: {path}")
    parts: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            parts.append(line.strip().upper())
    return "".join(parts)


def gather_long_segments(
    chroms: Sequence[str],
    base_len: int,
    segments_per_group: int,
    num_groups: Optional[int],
    mode: str,
    seed: int,
) -> List[Segment]:
    rng = random.Random(seed)
    segments: List[Segment] = []
    group_len = base_len * segments_per_group
    for raw in chroms:
        chrom = normalize_chrom(raw)
        seq = read_fasta(chrom)
        total_len = len(seq)
        block_count = total_len // base_len
        group_count = block_count // segments_per_group
        candidates: List[Segment] = []
        for idx in range(group_count):
            start = idx * group_len
            end = start + group_len - 1
            text = seq[start:start + group_len]
            candidates.append(Segment(chrom=chrom, start=start, end=end, text=text))
        if not candidates:
            continue
        if mode == "random" and num_groups is not None and num_groups < len(candidates):
            chosen = rng.sample(candidates, num_groups)
        else:
            chosen = candidates[:num_groups] if num_groups is not None else candidates
        segments.extend(chosen)
    return segments


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def compute_metrics(base: torch.Tensor, focus: torch.Tensor) -> Dict[str, float]:
    eps = 1e-8
    base = base + eps
    focus = focus + eps
    base = base / base.sum()
    focus = focus / focus.sum()

    diff = base - focus
    l1 = diff.abs().sum().item()
    l2 = torch.norm(diff, p=2).item()
    hellinger = torch.norm(torch.sqrt(base) - torch.sqrt(focus), p=2).item() / math.sqrt(2.0)

    m = 0.5 * (base + focus)
    kl_base_focus = torch.sum(base * torch.log(base / focus)).item()
    kl_base_m = torch.sum(base * torch.log(base / m)).item()
    kl_focus_m = torch.sum(focus * torch.log(focus / m)).item()
    js = 0.5 * (kl_base_m + kl_focus_m)

    return {
        "l1": l1,
        "l2": l2,
        "hellinger": hellinger,
        "js": js,
        "kl_base_focus": kl_base_focus,
    }


def compute_focus_probs_stream(
    tokens: torch.Tensor,
    adapter: LLMAdapter,
    inserter: FocusTokenInserter,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    tokens = tokens.to(device)
    total_len = tokens.size(0)
    if total_len == 0:
        raise ValueError("Token sequence empty.")

    state = inserter.init_inference_state()
    attention_mask = torch.zeros((1, 0), dtype=torch.long, device=device)
    focus_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
    past = None
    processed = 0
    last_logits: Optional[torch.Tensor] = None

    with torch.no_grad():
        while processed < total_len:
            chunk = tokens[processed : processed + chunk_size].unsqueeze(0)
            chunk_len = chunk.size(1)
            if chunk_len == 0:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, chunk_len), dtype=torch.long, device=device)],
                dim=1,
            )
            focus_mask = torch.cat(
                [focus_mask, torch.zeros((1, chunk_len), dtype=torch.bool, device=device)],
                dim=1,
            )
            outputs = adapter.model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :].squeeze(0)
            processed += chunk_len

            for _ in range(chunk_len):
                inserter.observe_token(state)
                insertion = inserter.maybe_insert_focus_tokens(state)
                if insertion:
                    focus_tokens = torch.tensor(
                        insertion.focus_token_ids,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(focus_tokens, dtype=torch.long)],
                        dim=1,
                    )
                    focus_flags = torch.ones_like(focus_tokens, dtype=torch.bool)
                    focus_mask = torch.cat([focus_mask, focus_flags], dim=1)
                    outputs = adapter.model(
                        input_ids=focus_tokens,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = adapter.replace_or_merge_kv(outputs.past_key_values, focus_mask[0])

    if last_logits is None:
        raise RuntimeError("No logits computed for focus-token stream.")
    return softmax_probs(last_logits)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare next-token distributions on 10k-length segments.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--chroms", type=str, required=True)
    parser.add_argument("--base-segment-length", type=int, default=1024)
    parser.add_argument("--segments-per-group", type=int, default=10)
    parser.add_argument("--segment-length", type=int, default=None, help="Final concatenated length. Defaults to base*segments_per_group.")
    parser.add_argument("--num-seqs", type=int, default=None, help="Number of 10k segments per chromosome.")
    parser.add_argument("--sample", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    chroms = [normalize_chrom(c) for c in args.chroms.split(",") if c.strip()]
    if not chroms:
        raise ValueError("Chromosome list empty.")

    group_len = args.segment_length or (args.base_segment_length * args.segments_per_group)
    segments = gather_long_segments(
        chroms,
        args.base_segment_length,
        args.segments_per_group,
        args.num_seqs,
        args.sample,
        args.seed,
    )
    if not segments:
        raise ValueError("No segments gathered; adjust parameters.")
    print(f"Total segments collected: {len(segments)}")

    cfg_path = args.run_dir / "configs" / "focus_config.generated.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_yaml(str(cfg_path))
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    focus_adapter = LLMAdapter.from_pretrained(cfg["model_name_or_path"], cfg, device=device)
    ckpt = args.run_dir / "checkpoints" / "focus_adapter.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Focus-token checkpoint missing: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    focus_adapter.focus_attention.load_state_dict(state["focus_attention"])
    focus_adapter.model.get_input_embeddings().load_state_dict(state["embedding"])
    focus_adapter.eval()

    base_cfg = dict(cfg)
    base_cfg["model_name_or_path"] = str(args.base_model)
    base_adapter = LLMAdapter.from_pretrained(base_cfg["model_name_or_path"], base_cfg, device=device)
    base_adapter.eval()

    inserter = FocusTokenInserter(cfg, focus_adapter.tokenizer)
    block_len = args.max_tokens or group_len
    stream_chunk = int(cfg.get("block_len", 1024))

    dataset = SegmentDataset(segments, focus_adapter.tokenizer)
    pad_id = focus_adapter.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        chroms = []
        starts = []
        ends = []
        raw_tokens: List[torch.Tensor] = []
        lengths = []
        for i, item in enumerate(batch):
            seq = item["input_ids"]
            length = seq.size(0)
            input_ids[i, :length] = seq
            attention_mask[i, :length] = 1
            chroms.append(item["chrom"])
            starts.append(item["start"])
            ends.append(item["end"])
            raw_tokens.append(seq)
            lengths.append(length)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "chrom": chroms,
            "start": torch.tensor(starts, dtype=torch.long, device=device),
            "end": torch.tensor(ends, dtype=torch.long, device=device),
            "raw_input_ids": raw_tokens,
            "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    totals = {"l1": 0.0, "l2": 0.0, "hellinger": 0.0, "js": 0.0, "kl_base_focus": 0.0}
    rows: List[str] = []
    per_chrom_rows: Dict[str, List[Tuple[int, int, Dict[str, float]]]] = defaultdict(list)
    per_chrom_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {key: 0.0 for key in totals})
    per_chrom_counts: Dict[str, int] = defaultdict(int)
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            base_out = base_adapter.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            valid_len = batch["attention_mask"].sum(dim=1)
            base_probs = []
            for i in range(batch["input_ids"].size(0)):
                idx = max(int(valid_len[i].item()) - 1, 0)
                base_probs.append(softmax_probs(base_out.logits[i, idx, :]))
            base_probs = torch.stack(base_probs, dim=0)

            focus_probs = []
            for tokens in batch["raw_input_ids"]:
                probs = compute_focus_probs_stream(tokens, focus_adapter, inserter, device, stream_chunk)
                focus_probs.append(probs)
            focus_probs = torch.stack(focus_probs, dim=0)

            for i in range(base_probs.size(0)):
                metrics = compute_metrics(base_probs[i], focus_probs[i])
                metrics_copy = {key: float(value) for key, value in metrics.items()}
                for k in totals:
                    totals[k] += metrics_copy[k]
                chrom = batch["chrom"][i]
                start = int(batch["start"][i].item())
                end = int(batch["end"][i].item())
                rows.append(
                    f"{chrom}\t{start}\t{end}\t"
                    f"{metrics_copy['l1']:.6f}\t{metrics_copy['l2']:.6f}\t"
                    f"{metrics_copy['hellinger']:.6f}\t{metrics_copy['js']:.6f}\t"
                    f"{metrics_copy['kl_base_focus']:.6f}"
                )
                per_chrom_rows[chrom].append((start, end, metrics_copy))
                chrom_totals = per_chrom_totals[chrom]
                for key in totals:
                    chrom_totals[key] += metrics_copy[key]
                per_chrom_counts[chrom] += 1
                count += 1

    if count == 0:
        print("No segments evaluated.")
        return 1

    averages = {k: totals[k] / count for k in totals}
    header = "chrom\tstart\tend\tL1\tL2\tHellinger\tJensen-Shannon\tKL(base||focus)"
    print(header)
    for row in rows:
        print(row)
    print(f"\nAverages over {count} segments:")
    print(
        f"L1={averages['l1']:.6f}, L2={averages['l2']:.6f}, "
        f"Hellinger={averages['hellinger']:.6f}, JS={averages['js']:.6f}, "
        f"KL(base||focus)={averages['kl_base_focus']:.6f}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for row in rows:
                fh.write(row + "\n")
            fh.write(
                "# averages\t"
                f"{averages['l1']:.6f}\t{averages['l2']:.6f}\t"
                f"{averages['hellinger']:.6f}\t{averages['js']:.6f}\t"
                f"{averages['kl_base_focus']:.6f}\n"
            )

    output_dir = args.output_dir or (args.run_dir / "outputs" / "GRCh38_10k")
    output_dir.mkdir(parents=True, exist_ok=True)
    header_fields = ["chrom", "start", "end", "L1", "L2", "Hellinger", "Jensen-Shannon", "KL(base||focus)"]
    for chrom, chrom_rows in per_chrom_rows.items():
        chrom_totals = per_chrom_totals[chrom]
        chrom_count = per_chrom_counts[chrom]
        chrom_avg = {key: (chrom_totals[key] / chrom_count) if chrom_count else float("nan") for key in totals}
        file_path = output_dir / f"{chrom}_10k_next_token_metrics.csv"
        with file_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header_fields)
            for start, end, metrics in chrom_rows:
                writer.writerow(
                    [
                        chrom,
                        start,
                        end,
                        f"{metrics['l1']:.6f}",
                        f"{metrics['l2']:.6f}",
                        f"{metrics['hellinger']:.6f}",
                        f"{metrics['js']:.6f}",
                        f"{metrics['kl_base_focus']:.6f}",
                    ]
                )
            writer.writerow(
                [
                    "AVERAGE",
                    "",
                    "",
                    f"{chrom_avg['l1']:.6f}",
                    f"{chrom_avg['l2']:.6f}",
                    f"{chrom_avg['hellinger']:.6f}",
                    f"{chrom_avg['js']:.6f}",
                    f"{chrom_avg['kl_base_focus']:.6f}",
                ]
            )
        print(f"Saved {chrom} metrics to {file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
