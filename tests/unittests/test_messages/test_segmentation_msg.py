import depthai as dai
import numpy as np
import pytest

from depthai_nodes import SegmentationMask


@pytest.fixture
def segmentation_mask():
    return SegmentationMask()


def test_segmentation_mask_initialization(segmentation_mask: SegmentationMask):
    assert np.array_equal(segmentation_mask.mask, np.array([]))
    assert segmentation_mask.transformation is None


def test_segmentation_mask_set_mask(segmentation_mask: SegmentationMask):
    mask_array = np.random.randint(-1, 256, (480, 640), dtype=np.int16)
    segmentation_mask.mask = mask_array
    assert np.array_equal(segmentation_mask.mask, mask_array)

    with pytest.raises(TypeError):
        segmentation_mask.mask = "not a numpy array"

    with pytest.raises(ValueError):
        segmentation_mask.mask = np.random.randint(
            -1, 256, (480, 640, 3), dtype=np.int16
        )

    with pytest.raises(ValueError):
        segmentation_mask.mask = np.random.randint(-1, 256, (480, 640), dtype=np.uint8)

    with pytest.raises(ValueError):
        segmentation_mask.mask = np.random.randint(-2, 256, (480, 640), dtype=np.int16)


def test_segmentation_mask_set_transformation(segmentation_mask: SegmentationMask):
    transformation = dai.ImgTransformation()
    segmentation_mask.transformation = transformation
    assert segmentation_mask.transformation == transformation

    with pytest.raises(TypeError):
        segmentation_mask.transformation = "not a dai.ImgTransformation"


def test_segmentation_mask_set_transformation_none(segmentation_mask: SegmentationMask):
    segmentation_mask.transformation = None
    assert segmentation_mask.transformation is None
