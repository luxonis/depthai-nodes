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

# The login shell activates /Users/hil/.hil, which is itself a virtual
# environment. Create the project environment from its base interpreter rather
# than nesting another environment inside HIL's one.
BASE_PYTHON="$(python3.12 -c 'import sys; print(sys._base_executable)')"
"$BASE_PYTHON" -m venv --clear --copies venv
VENV_PYTHON="$PWD/venv/bin/python"

# The HIL host has its own Python environment. Refuse to run unless this is
# specifically the project environment, not the pre-activated HIL environment.
"$VENV_PYTHON" -c '
import os
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.prefix={sys.prefix}")
print(f"sys.base_prefix={sys.base_prefix}")
expected_prefix = os.path.realpath(sys.argv[1])
actual_prefix = os.path.realpath(sys.prefix)
if actual_prefix != expected_prefix:
    raise SystemExit(
        f"Expected project virtual environment {expected_prefix}, got {actual_prefix}"
    )
' "$PWD/venv"

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e .
"$VENV_PYTHON" -m pip install -r requirements-dev.txt

# Install depthai with required indexes
"$VENV_PYTHON" -m pip install --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  ${LUXONIS_EXTRA_INDEX_URL:+--extra-index-url "$LUXONIS_EXTRA_INDEX_URL"} \
  "depthai==${DEPTHAI_VERSION}"

cd tests/end_to_end

source <("$VENV_PYTHON" setup_camera_ips.py)
export DEPTHAI_NODES_LEVEL=debug

# Verify the import in the exact interpreter that will execute the test.
"$VENV_PYTHON" -c 'import sys, depthai_nodes; print(sys.executable); print(depthai_nodes.__file__)'

"$VENV_PYTHON" -u main.py --platform "${PLATFORM}"
