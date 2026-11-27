from datetime import datetime, timedelta
from types import SimpleNamespace

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ApplyColormap, ApplyDepthColormap


@pytest.mark.parametrize("node_type", [ApplyColormap, ApplyDepthColormap])
def test_colorized_frame_preserves_source_metadata(node_type):
    source = dai.ImgFrame()
    source.setTimestamp(timedelta(seconds=2))
    source.setTimestampDevice(timedelta(seconds=12))
    source.setTimestampSystem(datetime(2026, 1, 2))
    source.setSequenceNum(9)
    source.setTransformation(dai.ImgTransformation(2, 2))

    output = node_type._build_output_frame(
        SimpleNamespace(_img_frame_type=dai.ImgFrame.Type.BGR888i),
        np.zeros((2, 2, 3), dtype=np.uint8),
        source,
    )

    assert output.getTimestamp() == source.getTimestamp()
    assert output.getTimestampDevice() == source.getTimestampDevice()
    assert output.getTimestampSystem() == source.getTimestampSystem()
    assert output.getSequenceNum() == source.getSequenceNum()
    assert output.getTransformation().getSize() == (2, 2)
