import depthai as dai
import pytest

from depthai_nodes import Keypoints


@pytest.fixture
def keypoints():
    return Keypoints()


def test_keypoints_initialization(keypoints: Keypoints):
    assert isinstance(keypoints.keypoints_list, dai.KeypointsList)
    assert keypoints.getKeypoints() == []
    assert keypoints.getEdges() == []
    assert keypoints.transformation is None


def test_keypoints_set_keypoints_list(keypoints: Keypoints):
    native = dai.KeypointsList()
    point1 = dai.Keypoint()
    point1.imageCoordinates = dai.Point3f(0.1, 0.2, 0.0)
    point2 = dai.Keypoint()
    point2.imageCoordinates = dai.Point3f(0.3, 0.4, 0.0)
    native.setKeypoints([point1, point2])
    native.setEdges([(0, 1)])
    keypoints.keypoints_list = native

    assert keypoints.keypoints_list is native
    assert keypoints.getKeypoints()[0].imageCoordinates.x == pytest.approx(0.1)
    assert keypoints.getEdges() == [[0, 1]]

    with pytest.raises(TypeError):
        keypoints.keypoints_list = "not a dai.KeypointsList"


def test_keypoints_set_transformation(keypoints: Keypoints):
    transformation = dai.ImgTransformation()
    keypoints.transformation = transformation
    assert keypoints.transformation == transformation

    with pytest.raises(TypeError):
        keypoints.transformation = "not a dai.ImgTransformation"


def test_keypoints_set_transformation_none(keypoints: Keypoints):
    keypoints.transformation = None
    assert keypoints.transformation is None
