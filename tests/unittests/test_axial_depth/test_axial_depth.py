"""Keep real SDK checks separate from parser tests that replace DepthAI classes."""

import os
import subprocess
import sys
from pathlib import Path


def test_axial_depth_native_contracts():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).with_name("checks.py")),
            "-q",
        ],
        cwd=root,
        env=dict(os.environ, DEPTHAI_TELEMETRY="0", PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
