"""Verify installed wheel bytes and run the deterministic host geometry fixture."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    """Check the real installed artifact, then retain its geometry evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("depthai-nodes")
    installed_root = Path(distribution.locate_file("depthai_nodes")).resolve()
    if not installed_root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Install the wheel into this environment without editable mode"
        )
    verified = 0
    with zipfile.ZipFile(args.wheel) as wheel:
        if any(name.startswith(("tests/", "examples/")) for name in wheel.namelist()):
            raise RuntimeError("The wheel includes repository test or example packages")
        for name in wheel.namelist():
            if name.endswith("/") or not (
                name.startswith("depthai_nodes/")
                or name.endswith(".dist-info/METADATA")
            ):
                continue
            installed = Path(distribution.locate_file(name))
            if installed.read_bytes() != wheel.read(name):
                raise RuntimeError(f"Installed bytes differ from the wheel at {name}")
            verified += 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(args.example.resolve()),
            "--report",
            str(args.report.resolve()),
        ],
        check=True,
        timeout=120,
    )
    report = json.loads(args.report.read_text())
    report["artifact"] = dict(
        revision=args.revision,
        wheel_name=args.wheel.name,
        wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        verified_installed_files=verified,
        installed_package_path=str(installed_root),
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["artifact"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
