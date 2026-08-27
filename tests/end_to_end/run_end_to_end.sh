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

case "$(uname -s)" in
  Darwin)
    HIL_VENV="venv"
    if [[ "$(uname -m)" == "arm64" ]]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    else
      eval "$(/usr/local/bin/brew shellenv)"
    fi
    PYTHON312="$(brew --prefix python@3.12)/bin/python3.12"
    uv venv --clear --seed --python "$PYTHON312" "$HIL_VENV"
    # shellcheck disable=SC1091
    source "$HIL_VENV/bin/activate"
    ;;
  Linux)
    python3.12 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    ;;
  *)
    echo "Other OS: $(uname -s)"
    exit 1
    ;;
esac


python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt

# Install depthai with required indexes
pip install --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  ${LUXONIS_EXTRA_INDEX_URL:+--extra-index-url "$LUXONIS_EXTRA_INDEX_URL"} \
  "depthai==${DEPTHAI_VERSION}"

cd tests/end_to_end

python setup_camera_ips.py > camera_ips.env
source ./camera_ips.env
export DEPTHAI_NODES_LEVEL=debug
python -u main.py --platform "${PLATFORM}"
