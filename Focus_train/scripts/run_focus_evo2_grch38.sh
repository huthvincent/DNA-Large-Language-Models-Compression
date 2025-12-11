#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_focus_evo2_grch38.py" "$@"
