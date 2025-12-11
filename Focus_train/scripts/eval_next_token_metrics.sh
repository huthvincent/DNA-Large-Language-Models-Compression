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

python "${ROOT_DIR}/scripts/eval_next_token_metrics.py" \
  --run-dir "${ROOT_DIR}/output/focus_runs/focus_chr1" \
  --base-model "${ROOT_DIR}/checkpoints/evo2_7b_nemo_flattened" \
  --chroms 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y \
  --segment-length 1024 \
  --num-seqs 500 \
  --sample random \
  --batch-size 8 \
  --output-dir "${ROOT_DIR}/output/focus_runs/focus_chr1/outputs/GRCh38"
