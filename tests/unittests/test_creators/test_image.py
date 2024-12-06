import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators import (
    create_image_message,
)


def test_valid_hwc_bgr():
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i


def test_valid_hwc_rgb():
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=False)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i


def test_valid_chw_bgr():
    image = np.random.randint(0, 256, (3, 480, 640), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i


def test_valid_chw_rgb():
    image = np.random.randint(0, 256, (3, 480, 640), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=False)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i


def test_valid_hwc_grayscale():
    image = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.GRAY8


def test_valid_chw_grayscale():
    image = np.random.randint(0, 256, (1, 480, 640), dtype=np.uint8)
    img_frame = create_image_message(image, is_bgr=True)

    assert isinstance(img_frame, dai.ImgFrame)
    assert img_frame.getWidth() == 640
    assert img_frame.getHeight() == 480
    assert img_frame.getType() == dai.ImgFrame.Type.GRAY8


def test_invalid_shape():
    image = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unexpected image shape. Expected CHW or HWC"):
        create_image_message(image, is_bgr=True)


def test_invalid_dtype():
    image = np.random.rand(480, 640, 3).astype(np.float32)
    with pytest.raises(
        ValueError, match="Expected int type, got <class 'numpy.float32'>."
    ):
        create_image_message(image, is_bgr=True)


def test_float_array():
    img = np.array([[[0.5, 0.5, 0.5]]])
    with pytest.raises(
        ValueError, match="Expected int type, got <class 'numpy.float64'>."
    ):
        create_image_message(img)
