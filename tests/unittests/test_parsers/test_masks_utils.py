import numpy as np
import pytest

from depthai_nodes.node.parsers.utils.masks_utils import process_single_mask


def test_mask_threshold_is_applied_after_bilinear_resize():
    protos = np.array([[[-3, 1], [-3, 1]]], dtype=np.float32)

    mask = process_single_mask(
        protos,
        np.array([1], dtype=np.float32),
        0.5,
        np.array([0.5, 0.5, 1, 1]),
        (2, 4),
    )

    # Resized logits are [-3, -2, 0, 1]. Thresholding before resizing
    # would incorrectly include the third column.
    np.testing.assert_array_equal(mask, [[0, 0, 0, 1], [0, 0, 0, 1]])
    assert mask.dtype == np.uint8


@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
def test_cropped_pixels_remain_background_at_different_thresholds(threshold):
    mask = process_single_mask(
        np.full((1, 2, 2), 2, dtype=np.float32),
        np.array([1], dtype=np.float32),
        threshold,
        np.array([0.75, 0.5, 0.5, 1]),
        (2, 2),
    )

    np.testing.assert_array_equal(mask, [[0, 1], [0, 1]])
