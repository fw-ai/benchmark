#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_init_venv.sh
source "${SCRIPT_DIR}/common_init_venv.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${REPO_ROOT}/.venv-sglang"

if [[ ! -x "${VENV}/bin/python" ]]; then
  bright_yellow python3 -m venv "${VENV}"
fi

bright_yellow "${VENV}/bin/python" -m pip install --upgrade pip wheel setuptools
bright_yellow "${VENV}/bin/python" -m pip install -r "${REPO_ROOT}/llm_bench/requirements.txt"
bright_yellow "${VENV}/bin/python" -m pip install aiohttp huggingface_hub matplotlib pyyaml "sglang[all]"
