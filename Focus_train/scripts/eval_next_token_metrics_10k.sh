#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/vendor:\
${ROOT_DIR}/vendor/bionemo/sub-packages/bionemo-core/src:\
${ROOT_DIR}/vendor/bionemo/sub-packages/bionemo-llm/src:\
${ROOT_DIR}/vendor/bionemo/sub-packages/bionemo-evo2/src:\
${ROOT_DIR}/vendor/bionemo/3rdparty/NeMo:\
${ROOT_DIR}/vendor/bionemo/3rdparty/Megatron-LM:\
${ROOT_DIR}:${PYTHONPATH:-}"

python "${ROOT_DIR}/scripts/eval_next_token_metrics_10k.py" \
  --run-dir "${ROOT_DIR}/output/focus_runs/focus_chr1" \
  --base-model "${ROOT_DIR}/checkpoints/evo2_7b_nemo_flattened" \
  --chroms 2 \
  --base-segment-length 1024 \
  --segments-per-group 6 \
  --segment-length 10240 \
  --num-seqs 2 \
  --sample sequential \
  --batch-size 2 \
  --output "${ROOT_DIR}/output/focus_runs/focus_chr1/outputs/GRCh38_10k/summary.tsv" \
  --output-dir "${ROOT_DIR}/output/focus_runs/focus_chr1/outputs/GRCh38_10k"
