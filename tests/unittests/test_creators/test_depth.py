import re

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.depth import create_depth_message

UINT16_MAX_VALUE = 65535
np.random.seed(0)


def test_not_numpy_array():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'list'>."):
        create_depth_message([1, 2, 3], "relative")


def test_wrong_literal_type():
    with pytest.raises(ValueError):
        create_depth_message(np.array([1, 2, 3]), "wrong")


def test_not_3d_input():
    with pytest.raises(ValueError, match="Expected 2D or 3D input, got 1D input."):
        create_depth_message(np.array([1, 2, 3]), "relative")


def test_wrong_input_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Unexpected image shape. Expected NHW or HWN, got (3, 1, 3)."),
    ):
        create_depth_message(
            np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]), "relative"
        )


def test_depth_limit_for_relative_depth():
    depth_map = np.random.rand(320, 640, 1)
    with pytest.raises(
        ValueError,
        match="Invalid depth limit: 1.0. For relative depth, depth limit must be equal to 0.",
    ):
        create_depth_message(depth_map, "relative", 1.0)


def test_no_depth_limit_for_metric_depth():
    depth_map = np.random.rand(320, 640, 1)
    with pytest.raises(
        ValueError,
        match="Invalid depth limit: 0.0. For metric depth, depth limit must be bigger than 0.",
    ):
        create_depth_message(depth_map, "metric")


def test_negative_depth_limit():
    depth_map = np.random.rand(320, 640, 1)
    with pytest.raises(
        ValueError,
        match="Invalid depth limit: -1.0. Depth limit must be bigger than 0.",
    ):
        create_depth_message(depth_map, "metric", -1.0)


def test_relative_depth_map():
    depth_map = np.random.rand(320, 640, 1)

    message = create_depth_message(depth_map, "relative")
    depth_map = depth_map[:, :, 0]

    assert isinstance(message, dai.ImgFrame)
    assert message.getType() == dai.ImgFrame.Type.RAW16
    assert message.getWidth() == 640
    assert message.getHeight() == 320

    frame = message.getFrame()
    assert frame.shape == depth_map.shape
    scaled_depth_map = (
        (depth_map - depth_map.min())
        / (depth_map.max() - depth_map.min())
        * UINT16_MAX_VALUE
    )
    scaled_depth_map = scaled_depth_map.astype(np.uint16)
    assert np.all(np.isclose(frame, scaled_depth_map))


def test_metric_depth_map():
    depth_map = np.random.rand(320, 640, 1)
    depth_limit = 10.0

    message = create_depth_message(depth_map, "metric", depth_limit)
    depth_map = depth_map[:, :, 0]
    depth_map = np.clip(depth_map, a_min=None, a_max=depth_limit)

    assert isinstance(message, dai.ImgFrame)
    assert message.getType() == dai.ImgFrame.Type.RAW16
    assert message.getWidth() == 640
    assert message.getHeight() == 320

    frame = message.getFrame()
    assert frame.shape == depth_map.shape
    scaled_depth_map = (depth_map - 0) / depth_limit * UINT16_MAX_VALUE
    scaled_depth_map = scaled_depth_map.astype(np.uint16)
    assert np.all(np.isclose(frame, scaled_depth_map))


def test_same_depth():
    depth_map = np.ones((320, 640, 1))
    message = create_depth_message(depth_map, "relative")

    assert isinstance(message, dai.ImgFrame)
    assert message.getType() == dai.ImgFrame.Type.RAW16
    assert message.getWidth() == 640
    assert message.getHeight() == 320
    frame = message.getFrame()
    assert np.all(np.isclose(frame, np.zeros((320, 640))))


if __name__ == "__main__":
    pytest.main()
