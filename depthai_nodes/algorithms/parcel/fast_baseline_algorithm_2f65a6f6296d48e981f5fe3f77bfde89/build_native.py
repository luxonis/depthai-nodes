"""Build the optional C++17 kernels for this platform; no Python build dependency."""
import os
import subprocess
import sys
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    output = root / ("_kernels.dylib" if sys.platform == "darwin" else "_kernels.so")
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O3",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(root / "kernels.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


if __name__ == "__main__":
    print(build())
