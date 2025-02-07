import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message.creators import (
    create_image_message,
)

IMAGE = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
IMAGE_GRAY = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)


def test_valid_hwc_bgr():
    img_frame = create_image_message(IMAGE, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i
    assert np.array_equal(img_frame.getCvFrame(), IMAGE)


def test_valid_hwc_rgb():
    create_image_message(IMAGE, is_bgr=False)


def test_valid_chw_bgr():
    image = IMAGE.transpose(2, 0, 1)
    create_image_message(image, is_bgr=True)


def test_valid_chw_rgb():
    image = IMAGE.transpose(2, 0, 1)
    create_image_message(image, is_bgr=False)


def test_valid_hwc_grayscale():
    img_frame = create_image_message(IMAGE_GRAY, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.GRAY8


def test_valid_chw_grayscale():
    image = IMAGE_GRAY.transpose(2, 0, 1)
    create_image_message(image, is_bgr=True)


def test_invalid_shape():
    image = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        create_image_message(image, is_bgr=True)


def test_invalid_dtype():
    image = IMAGE.astype(np.float32)
    with pytest.raises(ValueError):
        create_image_message(image, is_bgr=True)
