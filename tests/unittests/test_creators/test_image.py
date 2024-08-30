import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages.creators.image import create_image_message


def test_wrong_shape():
    img = np.random.random((10, 1, 10))
    with pytest.raises(
        ValueError, match="Unexpected image shape. Expected CHW or HWC, got"
    ):
        create_image_message(img)


def test_grayscale():
    img = (np.random.random((1, 100, 100)) * 255).astype(np.uint8)
    img_frame = create_image_message(img)

    assert img_frame.getType() == dai.ImgFrame.Type.GRAY8
    assert img_frame.getWidth() == 100
    assert img_frame.getHeight() == 100
    assert np.all(np.isclose(img_frame.getFrame(), img[0, :, :]))


def test_float_array():
    img = np.array([[[0.5, 0.5, 0.5]]])
    with pytest.raises(
        ValueError, match="Expected int type, got <class 'numpy.float64'>."
    ):
        create_image_message(img)


def test_bgr():
    img = (np.random.random((3, 100, 100)) * 255).astype(np.uint8)
    img_frame = create_image_message(img, False)

    assert img_frame.getType() == dai.ImgFrame.Type.BGR888i
    assert img_frame.getWidth() == 100
    assert img_frame.getHeight() == 100
    img = np.transpose(img, (1, 2, 0))
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    img = np.stack([img_b, img_g, img_r], axis=2)

    assert np.all(np.isclose(img_frame.getFrame(), img))


if __name__ == "__main__":
    pytest.main()
