# BioNeMo Evo-2 Focus Token Guide

Quick reference for preparing data, launching Focus Token fine-tuning, and running evaluation inside the BioNeMo container.

## Layout

- `focus/`: Core Focus Token modules (insertion, attention, adapters).
- `configs/focus_config.yaml`: Template config with `${PROJECT_ROOT}` placeholders.
- `scripts/`: Helper entrypoints (container helpers, GRCh38 data prep, metrics).
- `data/`, `output/`, `checkpoints/`: User-provided assets (not included in this repo).

## Environment

1. Build or pull the BioNeMo Evo-2 container (e.g., `bionemo-lora:2.6.3`).
2. Mount this repo into the container, export `PROJECT_ROOT` to the in-container path, and set `PYTHONPATH` to include the vendor BioNeMo packages shipped with the image.
3. (Optional) `make install` to install the Python requirements inside the container.

## Data Prep (GRCh38 example)

1. Split the reference FASTA by chromosome (run on host or in-container):
   ```bash
   python data/split_grch38_by_chr.py \
     --input data/GCA_000001405.15_GRCh38_no_alt_analysis_set.PanSN.fa.gz \
     --output-dir data/GRCh38
   ```
2. Generate training/validation JSONL and a run-specific config:
   ```bash
   python scripts/run_focus_evo2_grch38.py \
     --chroms 1,2,X \
     --segment-length 60000 \
     --focus-window 1024 \
     --model-name-or-path ${PROJECT_ROOT}/checkpoints/evo2_7b_nemo_flattened \
     --output-dir ${PROJECT_ROOT}/output/focus_runs \
     --tag demo_chr1_2_x \
     --dry-run
   ```
   Drop `--dry-run` to start training immediately.

## Training & Inference

- Train from a config:
  ```bash
  python focus/train.py --config configs/focus_config.yaml
  ```
- Autoregressive generation:
  ```bash
  python focus/infer.py --config configs/focus_config.yaml \
    --prompt "Protein design requires long-range reasoning."
  ```

## Evaluation

- Core metrics (perplexity + runtime deltas):
  ```bash
  python focus/metrics.py --config configs/focus_config.yaml
  ```
- Memory sweep by sequence length:
  ```bash
  python scripts/eval_memory.py --run-dir <focus_run_dir> --base-model <baseline_ckpt>
  ```

## Troubleshooting

- **Import errors for NeMo/Megatron**: confirm `PYTHONPATH` contains the BioNeMo vendor packages and `PROJECT_ROOT` is exported inside the container.
- **Missing checkpoints**: download Evo-2 weights into `checkpoints/` and point `model_name_or_path` accordingly.
- **OOM**: reduce `insert_every_n`, `segment-length`, or batch size; increase `gradient_accumulation_steps`.
- **Validation too slow**: lower `metrics.eval_tasks[].max_samples` or use shorter JSONL slices.
