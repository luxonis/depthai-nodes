#!/bin/sh
set -eu

example_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
manual_dir="$example_dir/../support_plane_calibration"
sh "$manual_dir/stage.sh" "$@"
cp "$example_dir/main.py" "$example_dir/oakapp.toml" "$1/"
cp "$example_dir/frontend/index.html" "$example_dir/frontend/app.js" "$1/frontend/"
