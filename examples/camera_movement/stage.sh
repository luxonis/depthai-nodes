#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: sh stage.sh <new-app-directory>" >&2
    exit 2
fi

example_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH='' cd -- "$example_dir/../.." && pwd)
mkdir "$1"
mkdir "$1/library"
cp "$example_dir/main.py" "$example_dir/oakapp.toml" "$1/"
cp -R "$example_dir/frontend" "$1/"
cp -R "$repo_dir/depthai_nodes" "$1/library/"
cp "$repo_dir/pyproject.toml" "$repo_dir/LICENSE" "$repo_dir/README.md" \
    "$repo_dir/requirements.txt" "$repo_dir/requirements-dev.txt" \
    "$repo_dir/requirements-box-measurement.txt" "$1/library/"
find "$1" -name __pycache__ -type d -prune -exec rm -r {} +
