import subprocess
import sys
from pathlib import Path
from typing import cast

import depthai as dai
import pytest

from depthai_nodes.node import BoxDimensioner
from tests.utils import OutputMock

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_box_dimensioner_build_names_the_extra_without_open3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)

    with pytest.raises(
        ImportError, match=r"pip install depthai-nodes\[box-measurement\]"
    ):
        output = cast(dai.Node.Output, OutputMock())
        BoxDimensioner().build(output, output, (64, 48))


@pytest.mark.parametrize(
    "statement",
    [
        "import depthai_nodes",
        "import depthai_nodes.node",
        "from depthai_nodes.node import *",
    ],
)
def test_package_imports_without_open3d(statement):
    code = f"import sys; sys.modules['open3d'] = None; {statement}"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
