#!/bin/bash

# Usage: ./entrypoint.sh <DEPTHAI_VERSION> <FLAGS...>

DEPTHAI_VERSION=$1
rm -rf venv
ls -l
python3 -m venv venv 
ls -l
source venv/bin/activate
export LC_ALL=en_US.UTF-8
locale

# Check if the DEPTHAI_VERSION is experimental
if [[ "$DEPTHAI_VERSION" == *"experimental"* ]]; then
    export EXPERIMENTAL_DEPTHAI=true
    echo $PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/tmp/depthai-core/build/bindings/python && echo $PYTHONPATH && hostname
    echo "DEPTHAI_VERSION is experimental. Skipping installation and setting EXPERIMENTAL_DEPTHAI=true."
else
    pip install --extra-index-url "$LUXONIS_EXTRA_INDEX_URL" depthai=="$DEPTHAI_VERSION"
fi

pip install -e .
pip install -r requirements-dev.txt

# Source camera IPs and run main script
source <(python3 tests/end_to_end/setup_camera_ips.py)
export HUBAI_TEAM_SLUG=$2
export HUBAI_API_KEY=$3
python3 tests/end_to_end/main.py -all
