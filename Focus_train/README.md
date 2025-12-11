# Focus Token Compression for BioNeMo Evo-2

Focus Tokens compress long-context decoding on the BioNeMo Evo-2 Hyena 7B model by injecting learned sparsity tokens, training a lightweight attention module, and pruning KV caches. This repository extracts the training, inference, and evaluation pieces into a GitHub-ready project that runs inside the BioNeMo 2.6 container.

## Requirements

- BioNeMo 2.6 container (e.g., `bionemo-lora:2.6.3`) with PyTorch ≥2.1 and CUDA ≥12.
- Access to Evo-2 Hyena checkpoints from NVIDIA NGC.
- One H100/A100-class GPU with ≥80 GB VRAM is recommended.
- Docker with NVIDIA runtime.

## Quick Start

1) **Launch the container**
```bash
docker run --gpus all -it \
  --name focus-train \
  -v /path/to/Focus_train:/workspace/focus-train \
  -v /path/to/output:/output \
  bionemo-lora:2.6.3
```

2) **Set environment variables (inside the container)**
```bash
cd /workspace/focus-train
export PROJECT_ROOT=/workspace/focus-train
export PYTHONPATH=/opt/bionemo/sub-packages/bionemo-core/src:\
/opt/bionemo/sub-packages/bionemo-llm/src:\
/opt/bionemo/sub-packages/bionemo-evo2/src:${PYTHONPATH}
```

3) **Install Python requirements (inside the container)**
```bash
pip install -r requirements.txt
```

## Data and Checkpoints

- **Checkpoints**: Download Evo-2 Hyena weights from NGC into `checkpoints/evo2_7b_nemo`:
  ```bash
  python - <<'PY'
  from pathlib import Path
  from bionemo.core.data.load import load
  target = Path("${PROJECT_ROOT}") / "checkpoints" / "evo2_7b_nemo"
  target.mkdir(parents=True, exist_ok=True)
  ckpt = load("evo2/7b-8k:1.0")
  for folder in ("context", "weights"):
      src = ckpt / folder
      if src.exists():
          (target / folder).mkdir(parents=True, exist_ok=True)
          for item in src.iterdir():
              dest = target / folder / item.name
              if not dest.exists():
                  item.replace(dest)
  print(f"Checkpoint staged at {target}")
  PY
  ```
- **Training data**: Not included. Place your FASTA/JSONL/text under `data/` and update `configs/focus_config.yaml` paths (they expand `${PROJECT_ROOT}`).

## Training

- **Direct fine-tuning** (uses `configs/focus_config.yaml`):
  ```bash
  python focus/train.py --config configs/focus_config.yaml
  ```
- **GRCh38 pipeline** (builds JSONL shards and a run config):
  ```bash
  python scripts/run_focus_evo2_grch38.py \
    --chroms 1,2,X \
    --segment-length 60000 \
    --focus-window 1024 \
    --model-name-or-path ${PROJECT_ROOT}/checkpoints/evo2_7b_nemo_flattened \
    --output-dir ${PROJECT_ROOT}/output/focus_runs \
    --tag demo_chr1_2_x
  ```
  Add `--dry-run` to inspect the plan without starting training.

## Inference and Evaluation

- **Autoregressive generation**
  ```bash
  python focus/infer.py --config configs/focus_config.yaml \
    --prompt "Protein design requires multi-scale reasoning."
  ```
- **Perplexity + runtime metrics**
  ```bash
  python focus/metrics.py --config configs/focus_config.yaml
  ```
- **Memory sweep**
  ```bash
  python scripts/eval_memory.py \
    --run-dir ${PROJECT_ROOT}/output/focus_runs/focus_demo \
    --base-model ${PROJECT_ROOT}/checkpoints/evo2_7b_nemo_flattened
  ```

## Tests

Lightweight checks that do not require proprietary weights:
```bash
python -m pytest tests/test_masks.py tests/test_kv_prune.py
```

## Project Layout

- `focus/` – Focus Token modules (insertion, attention, adapters, metrics, inference).
- `configs/focus_config.yaml` – Template config with `${PROJECT_ROOT}` placeholders.
- `scripts/` – Helper scripts (container wrappers, GRCh38 data prep, evaluation).
- `tools/` – Utility scripts (e.g., model export).
- `tests/` – Minimal unit tests for masking and KV pruning.

## Troubleshooting

- **Import errors for Megatron/NeMo**: re-check `PYTHONPATH` inside the container.
- **Checkpoint not found**: verify the Evo-2 weights are under `checkpoints/evo2_7b_nemo` and match `model_name_or_path`.
- **OOM during training**: lower `insert_every_n`, batch size, or increase `grad_accum`; reduce `segment-length` in the GRCh38 pipeline.
- **Slow validation**: decrease `metrics.eval_tasks[].max_samples` or use smaller JSONL slices.
