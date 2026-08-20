#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Usage:
#   ./mac_end_to_end.sh \
#     <BRANCH/depthai-nodes-version> \
#     <HUBAI_API_KEY> \
#     <HUBAI_TEAM_SLUG> \
#     <DEPTHAI_VERSION> \
#     <LUXONIS_EXTRA_INDEX_URL> \
#     <PLATFORM> \
#     [ADDITIONAL_PARAMETER...]
#
# Example:
#   ./mac_end_to_end.sh abc123 myteam 3.0.0 https://idx.xxx rvc4 --foo bar
#
# This maps to env:
#   LUXONIS_EXTRA_INDEX_URL
#   DEPTHAI_VERSION
#   HUBAI_TEAM_SLUG
#   HUBAI_API_KEY
#   FLAGS = "<ADDITIONAL_PARAMETER...>"
# ============================================

HUBAI_API_KEY="${1:-}"
HUBAI_TEAM_SLUG="${2:-}"
DEPTHAI_VERSION="${3:-}"
LUXONIS_EXTRA_INDEX_URL="${4:-}"
PLATFORM="${5:-}"
shift $(( $# >= 5 ? 5 : $# )) || true
ADDITIONAL_PARAMETER="${*:-}"

# ---- Basic validation
[[ -n "$HUBAI_API_KEY" ]]   || { echo "[!] HUBAI_API_KEY is required"; exit 2; }
[[ -n "$HUBAI_TEAM_SLUG" ]] || { echo "[!] HUBAI_TEAM_SLUG is required"; exit 2; }
[[ -n "$DEPTHAI_VERSION" ]] || { echo "[!] DEPTHAI_VERSION is required"; exit 2; }
[[ -n "$PLATFORM" ]]        || { echo "[!] PLATFORM is required (e.g., rvc4)"; exit 2; }

# ---- Compose FLAGS (kept for parity, even if not used directly below)
FLAGS="${ADDITIONAL_PARAMETER}"

# ---- Export env (for any subprocesses that read them)
export LUXONIS_EXTRA_INDEX_URL
export DEPTHAI_VERSION
export HUBAI_TEAM_SLUG
export HUBAI_API_KEY
export FLAGS
export DEPTHAI_NODES_LEVEL="debug"
export DEPTHAI_DEBUG="0"

readonly UV_REQUIRED_VERSION="0.12.5"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv ${UV_REQUIRED_VERSION} is required on the macOS test host" >&2
  exit 1
fi

UV_VERSION="$(uv --version)"
if [[ "$UV_VERSION" != "uv ${UV_REQUIRED_VERSION}" ]]; then
  echo "Expected uv ${UV_REQUIRED_VERSION}, found ${UV_VERSION}" >&2
  exit 1
fi

uv venv --managed-python --clear --python 3.12 venv
VENV_PYTHON="$PWD/venv/bin/python"

echo "Project virtual environment:"
"$VENV_PYTHON" -c 'import sys; print(f"executable: {sys.executable}"); print(f"prefix: {sys.prefix}"); print(f"base prefix: {sys.base_prefix}")'

uv pip install --python "$VENV_PYTHON" --upgrade pip
uv pip install --python "$VENV_PYTHON" -e .
uv pip install --python "$VENV_PYTHON" -r requirements-dev.txt

# Install depthai with required indexes
uv pip install --python "$VENV_PYTHON" --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  ${LUXONIS_EXTRA_INDEX_URL:+--extra-index-url "$LUXONIS_EXTRA_INDEX_URL"} \
  "depthai==${DEPTHAI_VERSION}"

cd tests/end_to_end

source <("$VENV_PYTHON" setup_camera_ips.py)
export DEPTHAI_NODES_LEVEL=debug
"$VENV_PYTHON" -u main.py --platform "${PLATFORM}"
