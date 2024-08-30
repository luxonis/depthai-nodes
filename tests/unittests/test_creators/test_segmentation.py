import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.segmentation import create_segmentation_message


def test_empty_array():
    with pytest.raises(ValueError, match="Expected 3D input, got 1D input."):
        create_segmentation_message(np.array([]))


def test_wrong_instance():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'int'>."):
        create_segmentation_message(1)


def test_wrong_dimension():
    with pytest.raises(
        ValueError, match="Expected 1 channel in the third dimension, got 3 channels."
    ):
        create_segmentation_message(np.random.rand(10, 10, 3))


def test_float_array():
    with pytest.raises(
        ValueError, match="Expected int type, got <class 'numpy.float64'>."
    ):
        create_segmentation_message(np.random.rand(10, 10, 1))


def test_complete_types():
    x = (np.random.rand(10, 10, 1) * 255).astype(np.uint8)
    message = create_segmentation_message(x)

    assert isinstance(message, dai.ImgFrame)
    assert message.getWidth() == x.shape[1]
    assert message.getHeight() == x.shape[0]
    assert message.getType() == dai.ImgFrame.Type.RAW8
    assert np.all(np.isclose(message.getFrame(), x[:, :, 0]))


if __name__ == "__main__":
    pytest.main()
