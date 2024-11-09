import numpy as np
import pytest

from depthai_nodes.ml.messages import SegmentationMask
from depthai_nodes.ml.messages.creators.segmentation import create_segmentation_message


def test_wrong_instance():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'int'>."):
        create_segmentation_message(1)


def test_empty_array():
    with pytest.raises(ValueError, match="Expected 2D input, got 1D input."):
        create_segmentation_message(np.array([]))

def test_float_array():
    with pytest.raises(
        ValueError, match="Unexpected mask type. Expected an array of integers, got float64."
    ):
        create_segmentation_message(np.random.rand(10, 10))


def test_complete_types():
    x = (np.random.rand(10, 10) * 255).astype(np.uint8)
    message = create_segmentation_message(x)

    assert isinstance(message, SegmentationMask)
    assert message.mask.shape[1] == x.shape[1]
    assert message.mask.shape[0] == x.shape[0]
    assert np.all(np.isclose(message.mask, x[:, :]))


if __name__ == "__main__":
    pytest.main()
