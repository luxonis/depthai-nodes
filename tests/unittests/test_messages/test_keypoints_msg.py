import depthai as dai
import pytest

from depthai_nodes.ml.messages import Keypoint, Keypoints


@pytest.fixture
def keypoint():
    return Keypoint()


@pytest.fixture
def keypoints():
    return Keypoints()


def test_keypoint_initialization(keypoint: Keypoint):
    assert keypoint.x is None
    assert keypoint.y is None
    assert keypoint.z == 0.0
    assert keypoint.confidence == -1.0


def test_keypoint_set_x(keypoint: Keypoint):
    keypoint.x = 0.5
    assert keypoint.x == 0.5

    with pytest.raises(TypeError):
        keypoint.x = "not a float"

    with pytest.raises(ValueError):
        keypoint.x = 1.5


def test_keypoint_set_y(keypoint: Keypoint):
    keypoint.y = 0.5
    assert keypoint.y == 0.5

    with pytest.raises(TypeError):
        keypoint.y = "not a float"

    with pytest.raises(ValueError):
        keypoint.y = 1.5


def test_keypoint_set_z(keypoint: Keypoint):
    keypoint.z = 0.5
    assert keypoint.z == 0.5

    with pytest.raises(TypeError):
        keypoint.z = "not a float"


def test_keypoint_set_confidence(keypoint: Keypoint):
    keypoint.confidence = 0.9
    assert keypoint.confidence == 0.9

    with pytest.raises(TypeError):
        keypoint.confidence = "not a float"

    with pytest.raises(ValueError):
        keypoint.confidence = 1.5


def test_keypoints_initialization(keypoints: Keypoints):
    assert keypoints.keypoints == []
    assert keypoints.transformation is None


def test_keypoints_set_keypoints(keypoints: Keypoints):
    kp1 = Keypoint()
    kp2 = Keypoint()
    keypoints_list = [kp1, kp2]
    keypoints.keypoints = keypoints_list
    assert keypoints.keypoints == keypoints_list

    with pytest.raises(TypeError):
        keypoints.keypoints = "not a list"

    with pytest.raises(ValueError):
        keypoints.keypoints = [kp1, "not a Keypoint"]


def test_keypoints_set_transformation(keypoints: Keypoints):
    transformation = dai.ImgTransformation()
    keypoints.transformation = transformation
    assert keypoints.transformation == transformation

    with pytest.raises(TypeError):
        keypoints.transformation = "not a dai.ImgTransformation"


def test_keypoints_set_transformation_none(keypoints: Keypoints):
    keypoints.transformation = None
    assert keypoints.transformation is None
