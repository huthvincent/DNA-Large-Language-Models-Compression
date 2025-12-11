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

python "${ROOT_DIR}/scripts/eval_focus_loss.py" \
  --run-dir "${ROOT_DIR}/output/focus_runs/focus_chr1" \
  --chroms 2 \
  --segment-length 60000 \
  --num-seqs 50 \
  --batch-size 1
