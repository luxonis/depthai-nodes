#!/bin/zsh
set -euo pipefail

# ============================================
# Usage:
#   ./run_end_to_end.zsh \
#     <HUBAI_API_KEY> \
#     <HUBAI_TEAM_SLUG> \
#     <DEPTHAI_VERSION> \
#     <LUXONIS_EXTRA_INDEX_URL> \
#     <PLATFORM> \
#     [ADDITIONAL_PARAMETER...]
# ============================================

HUBAI_API_KEY="${1:-}"
HUBAI_TEAM_SLUG="${2:-}"
DEPTHAI_VERSION="${3:-}"
LUXONIS_EXTRA_INDEX_URL="${4:-}"
PLATFORM="${5:-}"

if (( $# >= 5 )); then
    shift 5
else
    shift $#
fi

# Preserve any additional parameters for the test command
ADDITIONAL_PARAMETERS=("$@")
FLAGS="$*"

# ---- Basic validation

[[ -n "$HUBAI_API_KEY" ]] || {
    echo "[!] HUBAI_API_KEY is required"
    exit 2
}

[[ -n "$HUBAI_TEAM_SLUG" ]] || {
    echo "[!] HUBAI_TEAM_SLUG is required"
    exit 2
}

[[ -n "$DEPTHAI_VERSION" ]] || {
    echo "[!] DEPTHAI_VERSION is required"
    exit 2
}

[[ -n "$PLATFORM" ]] || {
    echo "[!] PLATFORM is required, for example: rvc4"
    exit 2
}

# ---- Export environment

export LUXONIS_EXTRA_INDEX_URL
export DEPTHAI_VERSION
export HUBAI_TEAM_SLUG
export HUBAI_API_KEY
export FLAGS
export DEPTHAI_NODES_LEVEL="debug"
export DEPTHAI_DEBUG="0"

# ---- Locate project root

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT"

# ---- Create and activate virtual environment
rm -rf venv
/opt/homebrew/bin/python3.12 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt

# ---- Install DepthAI

PIP_ARGUMENTS=(
    --upgrade
    --extra-index-url
    "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/"
)

if [[ -n "$LUXONIS_EXTRA_INDEX_URL" ]]; then
    PIP_ARGUMENTS+=(
        --extra-index-url
        "$LUXONIS_EXTRA_INDEX_URL"
    )
fi

python -m pip install \
    "${PIP_ARGUMENTS[@]}" \
    "depthai==${DEPTHAI_VERSION}"

# ---- Detect camera IPs

cd "$SCRIPT_DIR"

camera_env="$(python setup_camera_ips.py)" || {
    echo "[!] Failed to detect camera IPs"
    exit 1
}

# setup_camera_ips.py outputs export commands.
# Command substitution waits for Python to finish before eval executes them.
eval "$camera_env"

echo "RVC2_IP=${RVC2_IP:-}"
echo "RVC4_IP=${RVC4_IP:-}"

# ---- Run end-to-end tests

export DEPTHAI_NODES_LEVEL=debug

python -u main.py \
    --platform "$PLATFORM" \
    "${ADDITIONAL_PARAMETERS[@]}"