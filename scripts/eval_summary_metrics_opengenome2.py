#!/usr/bin/env python3
"""
Evaluate next-token distribution differences on OpenGenome2 batch data.

This mirrors eval_next_token_metrics.py but samples segments from a tar archive
of FASTA files (optionally gzipped). The default configuration samples 2000
random 1,024-token segments across all sequences in batch1.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from eval_next_token_metrics import (
    Segment,
    SegmentDataset,
    compute_metrics,
    softmax_probs,
)
from beacon.adapter import LLMAdapter
from beacon.token_injection import BeaconInserter
from beacon.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAR = ROOT / "data" / "OpenGenome2" / "batch1.tar"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare next-token distributions using segments sampled from OpenGenome2 batch1 FASTA files."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tar-path", type=Path, default=None, help="Tar archive containing FASTA/FASTA.GZ files.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing FASTA/FASTA.GZ files.")
    parser.add_argument("--segment-length", type=int, default=1024)
    parser.add_argument("--num-seqs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional TSV output summarizing all segments.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-sequence CSV exports.")
    return parser.parse_args(list(argv) if argv is not None else None)


def read_fasta_records(handle: io.TextIOBase, source_label: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []

    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header and parts:
                seq_id = f"{source_label}|{header}"
                records.append((seq_id, "".join(parts).upper()))
            header = line[1:].split()[0]
            parts = []
        else:
            parts.append(line.upper())

    if header and parts:
        seq_id = f"{source_label}|{header}"
        records.append((seq_id, "".join(parts).upper()))

    return records


def load_sequences_from_tar(tar_path: Path) -> List[Tuple[str, str]]:
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar archive not found: {tar_path}")

    sequences: List[Tuple[str, str]] = []
    with tarfile.open(tar_path, "r") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if not members:
            raise ValueError(f"No files found inside tar archive: {tar_path}")

        for member in members:
            name = member.name
            if not (name.endswith(".fasta") or name.endswith(".fasta.gz")):
                continue
            file_label = Path(name).name
            base_label = file_label.replace(".fasta.gz", "").replace(".fasta", "")
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            if name.endswith(".gz"):
                stream: io.TextIOBase = io.TextIOWrapper(gzip.GzipFile(fileobj=extracted), encoding="utf-8")
            else:
                stream = io.TextIOWrapper(extracted, encoding="utf-8")
            with stream as handle:
                sequences.extend(read_fasta_records(handle, base_label))

    if not sequences:
        raise ValueError(f"No FASTA records parsed from archive: {tar_path}")
    return sequences


def load_sequences_from_dir(input_dir: Path) -> List[Tuple[str, str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sequences: List[Tuple[str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if not (path.name.endswith(".fasta") or path.name.endswith(".fasta.gz")):
            continue
        base_label = path.with_suffix("").name.replace(".fasta", "")
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                sequences.extend(read_fasta_records(handle, base_label))
        else:
            with path.open("r", encoding="utf-8") as handle:
                sequences.extend(read_fasta_records(handle, base_label))

    if not sequences:
        raise ValueError(f"No FASTA records parsed from directory: {input_dir}")
    return sequences


def sample_segments(
    sequences: Sequence[Tuple[str, str]],
    segment_len: int,
    num_seqs: int,
    seed: int,
) -> List[Segment]:
    rng = random.Random(seed)
    eligible: List[Tuple[str, str]] = [(seq_id, seq) for seq_id, seq in sequences if len(seq) >= segment_len]
    if not eligible:
        raise ValueError("No sequences long enough for requested segment length.")

    total_possible = sum(len(seq) - segment_len + 1 for _, seq in eligible)
    target = min(num_seqs, total_possible)
    if target < num_seqs:
        print(f"Requested {num_seqs} segments but only {total_possible} distinct positions available; sampling {target}.")

    segments: List[Segment] = []
    seen_positions = set()
    max_attempts = target * 10
    attempts = 0

    while len(segments) < target and attempts < max_attempts:
        seq_id, seq = rng.choice(eligible)
        seq_len = len(seq)
        start = rng.randint(0, seq_len - segment_len)
        key = (seq_id, start)
        if key in seen_positions:
            attempts += 1
            continue
        seen_positions.add(key)
        end = start + segment_len - 1
        text = seq[start : start + segment_len]
        segments.append(Segment(chrom=seq_id, start=start, end=end, text=text))
        attempts += 1

    if len(segments) < target:
        raise RuntimeError(
            f"Unable to sample requested number of distinct segments. Requested {target}, obtained {len(segments)}."
        )

    return segments


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if args.input_dir and args.tar_path:
        raise ValueError("Specify only one of --input-dir or --tar-path.")
    if args.input_dir:
        source_sequences = load_sequences_from_dir(args.input_dir)
    else:
        tar_path = args.tar_path or DEFAULT_TAR
        source_sequences = load_sequences_from_tar(tar_path)

    segments = sample_segments(
        source_sequences,
        segment_len=args.segment_length,
        num_seqs=args.num_seqs,
        seed=args.seed,
    )
    print(f"Total segments collected: {len(segments)}")

    cfg_path = args.run_dir / "configs" / "beacon_config.generated.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_yaml(str(cfg_path))
    os.environ.setdefault("MOUNT_IN_CONTAINER", str(ROOT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ab_adapter = LLMAdapter.from_pretrained(cfg["model_name_or_path"], cfg, device=device)
    ckpt = args.run_dir / "checkpoints" / "beacon_adapter.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Beacon checkpoint missing: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    ab_adapter.beacon_attention.load_state_dict(state["beacon_attention"])
    ab_adapter.model.get_input_embeddings().load_state_dict(state["embedding"])
    ab_adapter.eval()

    base_cfg = dict(cfg)
    base_cfg["model_name_or_path"] = str(args.base_model)
    base_adapter = LLMAdapter.from_pretrained(base_cfg["model_name_or_path"], base_cfg, device=device)
    base_adapter.eval()

    inserter = BeaconInserter(cfg, ab_adapter.tokenizer)
    block_len = args.max_tokens or int(cfg.get("block_len", 1024))

    dataset = SegmentDataset(segments, ab_adapter.tokenizer, block_len)

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
        chroms = [item["chrom"] for item in batch]
        starts = torch.tensor([item["start"] for item in batch], device=device, dtype=torch.long)
        ends = torch.tensor([item["end"] for item in batch], device=device, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chrom": chroms,
            "start": starts,
            "end": ends,
        }

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    totals = {"l1": 0.0, "l2": 0.0, "hellinger": 0.0, "js": 0.0, "kl_base_ab": 0.0}
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

            updated, meta = inserter.prepare_training_batch(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
            )
            ab_out = ab_adapter.forward_with_beacon(
                updated["input_ids"],
                updated["attention_mask"],
                meta.beacon_mask,
                causal_mask=meta.causal_mask,
                block_map=meta.block_map,
                use_cache=False,
            )
            ab_probs = []
            for i in range(updated["input_ids"].size(0)):
                block_map = meta.block_map[i]
                orig_len = block_map[-1][1]
                idx = max(orig_len - 1, 0)
                ab_probs.append(softmax_probs(ab_out.logits[i, idx, :]))
            ab_probs = torch.stack(ab_probs, dim=0)

            for i in range(base_probs.size(0)):
                metrics = compute_metrics(base_probs[i], ab_probs[i])
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
                    f"{metrics_copy['kl_base_ab']:.6f}"
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
    header = "chrom\tstart\tend\tL1\tL2\tHellinger\tJensen-Shannon\tKL(base||AB)"
    print(header)
    for row in rows:
        print(row)
    print(f"\nAverages over {count} segments:")
    print(
        f"L1={averages['l1']:.6f}, L2={averages['l2']:.6f}, "
        f"Hellinger={averages['hellinger']:.6f}, JS={averages['js']:.6f}, "
        f"KL(base||AB)={averages['kl_base_ab']:.6f}"
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
                f"{averages['kl_base_ab']:.6f}\n"
            )

    default_dir = args.run_dir / "outputs" / "OpenGenome2"
    output_dir = args.output_dir or default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    header_fields = ["sequence_id", "start", "end", "L1", "L2", "Hellinger", "Jensen-Shannon", "KL(base||AB)"]
    for chrom, chrom_rows in per_chrom_rows.items():
        chrom_totals = per_chrom_totals[chrom]
        chrom_count = per_chrom_counts[chrom]
        chrom_avg = {key: (chrom_totals[key] / chrom_count) if chrom_count else float("nan") for key in totals}
        sanitized = chrom.replace("/", "_").replace("|", "_").replace(":", "_")
        file_path = output_dir / f"{sanitized}_next_token_metrics.csv"
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
                        f"{metrics['kl_base_ab']:.6f}",
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
                    f"{chrom_avg['kl_base_ab']:.6f}",
                ]
            )
        print(f"Saved {chrom} metrics to {file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
