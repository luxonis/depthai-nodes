"""Isolate native node checks from the repository's parser-node monkeypatches."""

import os
import subprocess
import sys
from pathlib import Path


def test_measure_anything_messages_conversion_and_native_nodes():
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, DEPTHAI_TELEMETRY="0", PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("checks.py"))],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
