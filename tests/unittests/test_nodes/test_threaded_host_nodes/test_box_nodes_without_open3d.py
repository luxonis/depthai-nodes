import sys
from typing import Any, cast

import depthai as dai
import pytest

from depthai_nodes.node import BoxMeasurer, SupportPlaneCalibrator
from tests.utils import OutputMock, PipelineMock


def test_support_plane_calibrator_build_names_the_extra_without_open3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    output = cast(dai.Node.Output, OutputMock())
    calibrator = cast(
        SupportPlaneCalibrator,
        PipelineMock().create(cast(Any, SupportPlaneCalibrator)),
    )

    with pytest.raises(
        ImportError, match=r'pip install "depthai-nodes\[box-measurement\]"'
    ):
        calibrator.build(output, output, (64, 48))


def test_box_measurer_build_names_the_extra_without_open3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    output = cast(dai.Node.Output, OutputMock())
    pipeline = PipelineMock()
    pipeline.getDefaultDevice().setPlatform(dai.Platform.RVC4)
    measurer = cast(BoxMeasurer, pipeline.create(cast(Any, BoxMeasurer)))

    with pytest.raises(
        ImportError, match=r'pip install "depthai-nodes\[box-measurement\]"'
    ):
        measurer.build(output, output, (64, 48))
