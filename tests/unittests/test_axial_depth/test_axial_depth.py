"""Keep real SDK checks separate from parser tests that replace DepthAI classes."""

import json
import os
import subprocess
import sys
from importlib.metadata import version
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


def test_deterministic_host_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[3]
    report = tmp_path / "host.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples/image_align_depth_correction/main.py"),
            "--report",
            str(report),
        ],
        cwd=root,
        env=dict(os.environ, DEPTHAI_TELEMETRY="0", PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(report.read_text())
    assert result["sdk_version"] == version("depthai")
    assert result["source_z_retained_by_image_align"]
    assert result["correction"]["correction_enabled"]
    assert result["correction"]["output_axial_convention"] == "target_camera_axial_z"
