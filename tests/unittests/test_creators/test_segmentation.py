import numpy as np
import pytest

from depthai_nodes.ml.messages import SegmentationMask
from depthai_nodes.ml.messages.creators.segmentation import create_segmentation_message


def test_valid_input():
    mask = np.random.randint(0, 256, (480, 640), dtype=np.int16)
    message = create_segmentation_message(mask)

    assert isinstance(message, SegmentationMask)
    assert message.mask.shape == (480, 640)
    assert message.mask.dtype == np.int16


def test_invalid_type():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'list'>."):
        create_segmentation_message([[0, 1], [2, 3]])


def test_invalid_shape():
    mask = np.random.randint(0, 256, (480, 640, 3), dtype=np.int16)
    with pytest.raises(ValueError, match="Expected 2D input, got 3D input."):
        create_segmentation_message(mask)


def test_invalid_dtype():
    mask = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected int16 input, got uint8."):
        create_segmentation_message(mask)


def test_empty_mask():
    mask = np.empty((0, 0), dtype=np.int16)
    message = create_segmentation_message(mask)

    assert isinstance(message, SegmentationMask)
    assert message.mask.shape == (0, 0)
    assert message.mask.dtype == np.int16
