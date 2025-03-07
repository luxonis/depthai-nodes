#!/bin/bash

# Usage: ./run_e2e_tests.sh <DEPTHAI_VERSION> <FLAGS...>

DEPTHAI_VERSION=$1
rm -rf venv
python3 -m venv venv 
source venv/bin/activate
export LC_ALL=en_US.UTF-8

# Check if the DEPTHAI_VERSION is experimental
if [[ "$DEPTHAI_VERSION" == *"experimental"* ]]; then
    export EXPERIMENTAL_DEPTHAI=true
    export PYTHONPATH=$PYTHONPATH:/tmp/depthai-core/build/bindings/python
    echo "DEPTHAI_VERSION is experimental. Skipping installation and setting EXPERIMENTAL_DEPTHAI=true."
else
    LUXONIS_EXTRA_INDEX_URL=$4
    pip install --extra-index-url "$LUXONIS_EXTRA_INDEX_URL" depthai=="$DEPTHAI_VERSION"
fi

pip install -e .
pip install -r requirements-dev.txt

# Source camera IPs and run main script
cd tests/end_to_end
source <(python3 setup_camera_ips.py)
export HUBAI_TEAM_SLUG=$2
export HUBAI_API_KEY=$3
export DISPLAY=:99
python3 main.py --platform RVC4 --depthai-nodes-version $5