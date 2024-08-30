import re

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.thermal import create_thermal_message


def test_dimension():
    with pytest.raises(ValueError):
        create_thermal_message(np.array([[0], [1], [1]]))


def test_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Unexpected image shape. Expected CHW or HWC, got (2, 2, 2)."),
    ):
        create_thermal_message(np.array([[[0, 1], [1, 1]], [[0, 1], [1, 1]]]))


def test_float():
    with pytest.raises(ValueError, match="Expected integer values, got float."):
        create_thermal_message(np.array([[[0.3, 0.17], [0.5, 0.1]]]))


def test_negative():
    with pytest.raises(
        ValueError, match="All values of thermal_image have to be non-negative."
    ):
        create_thermal_message(np.array([[[0, 1], [1, -1]]]))


def test_return():
    image = np.array([[[5, 7, 8], [4, 10, 5]]])
    imgFrame = create_thermal_message(image)
    assert imgFrame.getType() == dai.ImgFrame.Type.RAW16
    assert imgFrame.getWidth() == 3
    assert imgFrame.getHeight() == 2
    assert np.all(np.isclose(imgFrame.getFrame(), image))


if __name__ == "__main__":
    pytest.main()
