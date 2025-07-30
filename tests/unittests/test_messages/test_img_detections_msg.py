import depthai as dai
import numpy as np
import pytest

from depthai_nodes import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoint,
    Keypoints,
)


@pytest.fixture
def img_detection_extended():
    return ImgDetectionExtended()


@pytest.fixture
def img_detections_extended():
    return ImgDetectionsExtended()


def test_img_detection_extended_initialization(
    img_detection_extended: ImgDetectionExtended,
):
    assert img_detection_extended.confidence == -1.0
    assert img_detection_extended.label == -1
    assert img_detection_extended.label_name == ""


def test_img_detection_extended_set_rotated_rect(
    img_detection_extended: ImgDetectionExtended,
):
    rect = (0.5, 0.5, 0.2, 0.2, 45.0)
    img_detection_extended.rotated_rect = rect
    assert np.allclose(img_detection_extended.rotated_rect.center.x, 0.5, atol=1e-3)
    assert np.allclose(img_detection_extended.rotated_rect.center.y, 0.5, atol=1e-3)
    assert np.allclose(img_detection_extended.rotated_rect.size.width, 0.2, atol=1e-3)
    assert np.allclose(img_detection_extended.rotated_rect.size.height, 0.2, atol=1e-3)
    assert np.allclose(img_detection_extended.rotated_rect.angle, 45.0, atol=1e-3)


def test_img_detection_extended_set_confidence(
    img_detection_extended: ImgDetectionExtended,
):
    img_detection_extended.confidence = 0.9
    assert img_detection_extended.confidence == 0.9

    img_detection_extended.confidence = 1.05
    assert img_detection_extended.confidence == 1.0
    assert isinstance(img_detection_extended.confidence, float)

    with pytest.raises(TypeError):
        img_detection_extended.confidence = "not a float"

    with pytest.raises(ValueError):
        img_detection_extended.confidence = 1.5


def test_img_detection_extended_set_label(img_detection_extended: ImgDetectionExtended):
    img_detection_extended.label = 1
    assert img_detection_extended.label == 1

    with pytest.raises(TypeError):
        img_detection_extended.label = "not an int"


def test_img_detection_extended_set_label_name(
    img_detection_extended: ImgDetectionExtended,
):
    img_detection_extended.label_name = "cat"
    assert img_detection_extended.label_name == "cat"

    with pytest.raises(TypeError):
        img_detection_extended.label_name = 123


def test_img_detection_extended_set_keypoints(
    img_detection_extended: ImgDetectionExtended,
):
    keypoints = []
    kp1 = Keypoint()
    kp1.x = 0.1
    kp1.y = 0.2
    kp2 = Keypoint()
    kp2.x = 0.3
    kp2.y = 0.4
    keypoints.append(kp1)
    keypoints.append(kp2)
    kpts = Keypoints()
    kpts.keypoints = keypoints
    img_detection_extended.keypoints = kpts
    for i, kp in enumerate(img_detection_extended.keypoints):
        assert kp.x == keypoints[i].x
        assert kp.y == keypoints[i].y

    with pytest.raises(TypeError):
        img_detection_extended.keypoints = "not a Keypoints object"


def test_img_detections_extended_initialization(
    img_detections_extended: ImgDetectionsExtended,
):
    assert img_detections_extended.detections == []
    assert isinstance(img_detections_extended.masks, np.ndarray)
    assert img_detections_extended.transformation is None


def test_img_detections_extended_set_detections(
    img_detections_extended: ImgDetectionsExtended,
):
    detection1 = ImgDetectionExtended()
    detection2 = ImgDetectionExtended()
    detections_list = [detection1, detection2]
    img_detections_extended.detections = detections_list
    assert img_detections_extended.detections == detections_list

    with pytest.raises(TypeError):
        img_detections_extended.detections = "not a list"

    with pytest.raises(TypeError):
        img_detections_extended.detections = [detection1, "not an ImgDetectionExtended"]


def test_img_detections_extended_set_masks(
    img_detections_extended: ImgDetectionsExtended,
):
    masks = np.random.randint(0, 256, (480, 640), dtype=np.int16)
    img_detections_extended.masks = masks
    assert np.array_equal(img_detections_extended.masks, masks)

    with pytest.raises(TypeError):
        img_detections_extended.masks = "not a numpy array"

    with pytest.raises(ValueError):
        img_detections_extended.masks = np.random.randint(
            0, 256, (480, 640, 3), dtype=np.int16
        )

    with pytest.raises(ValueError):
        img_detections_extended.masks = np.random.randint(
            0, 256, (480, 640), dtype=np.uint8
        )

    with pytest.raises(ValueError):
        img_detections_extended.masks = np.random.randint(
            -2, 256, (480, 640), dtype=np.int16
        )


def test_img_detections_extended_set_transformation(
    img_detections_extended: ImgDetectionsExtended,
):
    transformation = dai.ImgTransformation()
    img_detections_extended.transformation = transformation
    assert img_detections_extended.transformation == transformation

    with pytest.raises(TypeError):
        img_detections_extended.transformation = "not a dai.ImgTransformation"


def test_img_detections_extended_set_transformation_none(
    img_detections_extended: ImgDetectionsExtended,
):
    img_detections_extended.transformation = None
    assert img_detections_extended.transformation is None
