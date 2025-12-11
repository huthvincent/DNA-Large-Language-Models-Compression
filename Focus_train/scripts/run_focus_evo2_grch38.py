#!/usr/bin/env python3
"""
Focus Token fine-tuning entrypoint for BioNeMo Evo-2 on GRCh38 subsets.

Pipeline:
1) Load chr*.fasta files from data/GRCh38 and slice them into fixed-length DNA segments.
2) Write train/val JSONL shards (data/focus_grch38_*.jsonl).
3) Generate a one-off config from configs/focus_config.yaml.
4) Optionally dry-run to inspect the plan, or launch focus/train.py directly.

Example:
python scripts/run_focus_evo2_grch38.py \\
    --chroms 22 \\
    --model-name-or-path checkpoints/evo2_7b_nemo_flattened \\
    --output-dir output/focus_runs \\
    --tag demo_chr22 \\
    --max-steps 10 \\
    --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
GRCH38_DIR = DATA_ROOT / "GRCh38"
BASE_CONFIG = ROOT / "configs" / "focus_config.yaml"

CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GRCh38 DNA segments and launch Focus Token fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chroms",
        type=str,
        default="canonical",
        help="Chromosome list like '1,2,22' or 'X,Y'; use 'canonical' for chr1-22,chrX,chrY,chrM.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=50_000,
        help="Bases per training sample.",
    )
    parser.add_argument(
        "--focus-window",
        type=int,
        default=1024,
        help="Focus token window/block length written into the generated config.",
    )
    parser.add_argument(
        "--min-window-multiples",
        type=int,
        default=20,
        help="Segments shorter than focus_window * this value are dropped.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Fraction of samples routed to validation.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run tag; defaults to chrom list + timestamp.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate JSONL shards even if they already exist.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Evo-2 checkpoint path; falls back to $EVO2_MODEL_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for the generated run (configs/output/checkpoints).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs; forced to 1 when --max-steps is set.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step cap for smoke tests.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples (debug use).",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=512,
        help="Validation sample cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without launching training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    return parser.parse_args(argv)


def parse_chroms(expr: str) -> List[str]:
    expr = expr.strip()
    if not expr or expr.lower() == "canonical":
        return CANONICAL_CHROMS[:]
    result: List[str] = []
    for item in expr.split(","):
        token = item.strip()
        if not token:
            continue
        if not token.lower().startswith("chr"):
            token = f"chr{token.upper()}"
        token = token.replace("CHR", "chr")
        if token not in CANONICAL_CHROMS:
            raise ValueError(f"Unsupported chromosome identifier: {item}")
        result.append(token)
    if not result:
        raise ValueError("No chromosomes parsed from --chroms.")
    return result


def chrom_tag(chroms: Sequence[str]) -> str:
    if len(chroms) == len(CANONICAL_CHROMS) and sorted(chroms) == CANONICAL_CHROMS:
        return "canonical"
    return "_".join(chroms)


def read_fasta_sequence(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith(">"):
                continue
            seq = line.strip().upper()
            if seq:
                yield seq


def iter_segments(
    fasta_files: Sequence[Path],
    segment_length: int,
    min_length: int,
) -> Iterator[str]:
    buffer: List[str] = []
    buffered = 0

    def flush(force_final: bool = False) -> Iterator[str]:
        nonlocal buffer, buffered
        if not buffer:
            return iter(())
        joined = "".join(buffer)
        produced: List[str] = []
        idx = 0
        total = len(joined)
        while total - idx >= segment_length:
            produced.append(joined[idx : idx + segment_length])
            idx += segment_length
        remainder = joined[idx:]
        if force_final and len(remainder) >= min_length:
            produced.append(remainder)
            remainder = ""
        buffer = [remainder] if remainder else []
        buffered = len(remainder)
        return iter(produced)

    for fasta in fasta_files:
        for chunk in read_fasta_sequence(fasta):
            buffer.append(chunk)
            buffered += len(chunk)
            if buffered >= segment_length:
                for segment in flush():
                    yield segment

    for segment in flush(force_final=True):
        yield segment


def prepare_datasets(
    chroms: Sequence[str],
    segment_length: int,
    min_length: int,
    val_ratio: float,
    tag: str,
    overwrite: bool,
    seed: int,
) -> Tuple[Path, Path, Dict[str, float]]:
    rng = random.Random(seed)
    train_path = DATA_ROOT / f"focus_grch38_{tag}_train.jsonl"
    val_path = DATA_ROOT / f"focus_grch38_{tag}_val.jsonl"

    if not overwrite and train_path.exists() and val_path.exists():
        summary = {
            "train_samples": sum(1 for _ in train_path.open("r", encoding="utf-8")),
            "val_samples": sum(1 for _ in val_path.open("r", encoding="utf-8")),
            "train_avg_len": None,
            "val_avg_len": None,
            "skipped": 0,
        }
        return train_path, val_path, summary

    fasta_files = [GRCH38_DIR / f"{chrom}.fasta" for chrom in chroms]
    missing = [str(p) for p in fasta_files if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Missing or empty FASTA files; generate them with data/split_grch38_by_chr.py:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    train_stats = {"count": 0, "total": 0}
    val_stats = {"count": 0, "total": 0}
    skipped = 0

    with train_path.open("w", encoding="utf-8") as train_fh, val_path.open("w", encoding="utf-8") as val_fh:
        for segment in iter_segments(fasta_files, segment_length, min_length):
            if len(segment) < min_length:
                skipped += 1
                continue
            record = json.dumps({"text": segment}, ensure_ascii=False)
            if rng.random() < val_ratio:
                val_fh.write(record + "\n")
                val_stats["count"] += 1
                val_stats["total"] += len(segment)
            else:
                train_fh.write(record + "\n")
                train_stats["count"] += 1
                train_stats["total"] += len(segment)

    summary = {
        "train_samples": train_stats["count"],
        "val_samples": val_stats["count"],
        "train_avg_len": round(train_stats["total"] / max(train_stats["count"], 1), 2),
        "val_avg_len": round(val_stats["total"] / max(val_stats["count"], 1), 2),
        "skipped": skipped,
    }
    return train_path, val_path, summary


def resolve_model_path(arg: Optional[str]) -> Path:
    candidate = arg or os.environ.get("EVO2_MODEL_PATH")
    if not candidate:
        raise ValueError("Specify the model via --model-name-or-path or EVO2_MODEL_PATH.")
    model_path = Path(os.path.expandvars(os.path.expanduser(candidate)))
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    return model_path


def load_base_config() -> Dict:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG}")
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_run_dirs(root: Path, tag: str) -> Dict[str, Path]:
    run_dir = root / f"focus_{tag}"
    dirs = {
        "root": run_dir,
        "configs": run_dir / "configs",
        "outputs": run_dir / "outputs",
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def update_config(
    base_cfg: Dict,
    model_path: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    run_dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> Dict:
    cfg = dict(base_cfg)  # shallow copy

    cfg["model_name_or_path"] = str(model_path)
    cfg["block_len"] = int(args.focus_window)

    paths = cfg.setdefault("paths", {})
    paths["data_dir"] = str(run_dirs["root"] / "data")
    paths["output_dir"] = str(run_dirs["outputs"])
    paths["checkpoint_dir"] = str(run_dirs["checkpoints"])
    paths["log_dir"] = str(run_dirs["logs"])

    train_cfg = cfg.setdefault("train", {})
    train_cfg["epochs"] = args.epochs if not args.max_steps else 1
    train_cfg["lr"] = args.learning_rate
    train_cfg["batch_size"] = args.per_device_train_batch_size
    train_cfg["grad_accum"] = args.gradient_accumulation_steps
    train_cfg["random_ratio_sampling"] = False
    if args.max_steps:
        train_cfg["steps"] = args.max_steps
    else:
        train_cfg.pop("steps", None)
    if args.max_train_samples:
        train_cfg["max_train_samples"] = args.max_train_samples
    else:
        train_cfg.pop("max_train_samples", None)
    dataset_cfg = train_cfg.setdefault("dataset", {})
    dataset_cfg.update(
        {
            "loader": "jsonl",
            "path": str(train_jsonl),
            "text_field": "text",
            "split": "train",
            "validation_fraction": 0.0,
        }
    )

    metrics_cfg = cfg.setdefault("metrics", {})
    eval_tasks = metrics_cfg.setdefault("eval_tasks", [{}])
    eval_tasks[0].update(
        {
            "name": "perplexity",
            "loader": "jsonl",
            "path": str(val_jsonl),
            "split": "train",
            "text_field": "text",
            "max_samples": args.eval_max_samples,
        }
    )
    metrics_cfg["baseline_model_name_or_path"] = str(model_path)
    metrics_cfg["results_path"] = str(run_dirs["outputs"] / "results.json")

    cfg.setdefault("bionemo", {})
    cfg["bionemo"]["use_te"] = True

    cfg["insert_every_n"] = cfg.get("insert_every_n", 100)
    cfg["focus_tokens_per_block"] = cfg.get("focus_tokens_per_block", 1)
    cfg["condense_ratio"] = cfg.get("condense_ratio", 100)

    return cfg


def ensure_pythonpath(env: Dict[str, str]) -> None:
    parts = [
        str(ROOT / "vendor"),
        str(ROOT / "vendor" / "bionemo" / "sub-packages" / "bionemo-core" / "src"),
        str(ROOT / "vendor" / "bionemo" / "sub-packages" / "bionemo-llm" / "src"),
        str(ROOT / "vendor" / "bionemo" / "sub-packages" / "bionemo-evo2" / "src"),
        str(ROOT / "vendor" / "bionemo" / "3rdparty" / "NeMo"),
        str(ROOT / "vendor" / "bionemo" / "3rdparty" / "Megatron-LM"),
        str(ROOT),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = ":".join(parts)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    chroms = parse_chroms(args.chroms)
    tag = args.tag or f"{chrom_tag(chroms)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    min_required = args.min_window_multiples * args.focus_window
    if args.segment_length < min_required:
        raise ValueError(
            f"segment-length ({args.segment_length}) must be >= focus_window({args.focus_window}) * "
            f"min_window_multiples({args.min_window_multiples}) = {min_required}"
        )

    model_path = resolve_model_path(args.model_name_or_path)
    train_jsonl, val_jsonl, summary = prepare_datasets(
        chroms,
        args.segment_length,
        min_required,
        args.val_ratio,
        tag,
        args.overwrite,
        args.seed,
    )

    run_dirs = build_run_dirs(args.output_dir.expanduser(), tag)
    base_cfg = load_base_config()
    updated_cfg = update_config(base_cfg, model_path, train_jsonl, val_jsonl, run_dirs, args)
    config_path = run_dirs["configs"] / "focus_config.generated.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(updated_cfg, fh, sort_keys=False, allow_unicode=True)

    info_lines = [
        "========== Focus Token prep complete ==========",
        f"Chromosomes: {', '.join(chroms)}",
        f"Segment length: {args.segment_length} bp (min required {min_required} bp)",
        f"Train samples: {summary['train_samples']} (avg {summary['train_avg_len']} bp)",
        f"Validation samples: {summary['val_samples']} (avg {summary['val_avg_len']} bp)",
        f"Skipped short fragments: {summary['skipped']}",
        f"Generated config: {config_path}",
        f"Model weights: {model_path}",
        f"Run directory: {run_dirs['root']}",
    ]
    print("\n".join(info_lines))

    train_cmd = [
        sys.executable,
        str(ROOT / "focus" / "train.py"),
        "--config",
        str(config_path),
    ]
    print("\nTraining command:\n  " + " ".join(train_cmd))

    if args.dry_run:
        print("\n[DRY-RUN] Skipping training; data and config are ready.")
        return 0

    env = os.environ.copy()
    ensure_pythonpath(env)
    start = time.time()
    result = subprocess.run(train_cmd, cwd=str(ROOT), env=env)
    duration = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}. Check logs.")

    metrics_path = run_dirs["outputs"] / "train_metrics.json"
    final_loss = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            try:
                metrics = json.load(fh)
                final_loss = metrics.get("loss")
            except json.JSONDecodeError:
                final_loss = None

    print("\n========== Focus Token training complete ==========")
    print(f"Duration: {duration/60:.2f} minutes")
    if final_loss is not None:
        print(f"Final loss: {final_loss}")
    print(f"Checkpoint directory: {run_dirs['checkpoints']}")
    print(f"Outputs directory: {run_dirs['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
