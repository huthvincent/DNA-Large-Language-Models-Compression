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

python "${ROOT_DIR}/scripts/eval_memory.py" \
  --run-dir "${ROOT_DIR}/output/focus_runs/focus_chr1" \
  --base-model "${ROOT_DIR}/checkpoints/evo2_7b_nemo_flattened" \
  --lengths "1000,2000,4000,8000, 10000,20000,30000,40000,50000,60000,70000,80000" \
  --chunk-size 1024 \
  --output "${ROOT_DIR}/output/focus_runs/focus_chr1/outputs/memory/memory.csv" \
  --save-fasta
