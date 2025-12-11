#!/usr/bin/env bash
set -euo pipefail

python /workspace/bionemo-framework/workspace/focus/infer.py --config /workspace/bionemo-framework/workspace/configs/focus_config.yaml --prompt "Biology enables new drug discovery by modeling protein folding with long contexts."
