#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_init_venv.sh
source "${SCRIPT_DIR}/common_init_venv.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${REPO_ROOT}/.venv-fireworks"

if [[ ! -x "${VENV}/bin/python" ]]; then
  bright_yellow python3 -m venv --system-site-packages "${VENV}"
fi
