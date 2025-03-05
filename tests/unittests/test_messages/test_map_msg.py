import depthai as dai
import numpy as np
import pytest

from depthai_nodes import Map2D


@pytest.fixture
def map2d():
    return Map2D()


def test_map2d_initialization(map2d: Map2D):
    assert np.array_equal(map2d.map, np.array([]))
    assert map2d.width is None
    assert map2d.height is None
    assert map2d.transformation is None


def test_map2d_set_map(map2d: Map2D):
    map_array = np.random.rand(480, 640).astype(np.float32)
    map2d.map = map_array
    assert np.array_equal(map2d.map, map_array)
    assert map2d.width == 640
    assert map2d.height == 480

    with pytest.raises(TypeError):
        map2d.map = "not a numpy array"

    with pytest.raises(ValueError):
        map2d.map = np.random.rand(480, 640, 3).astype(np.float32)

    with pytest.raises(ValueError):
        map2d.map = np.random.rand(480, 640).astype(np.float64)


def test_map2d_set_transformation(map2d: Map2D):
    transformation = dai.ImgTransformation()
    map2d.transformation = transformation
    assert map2d.transformation == transformation

    with pytest.raises(TypeError):
        map2d.transformation = "not a dai.ImgTransformation"


def test_map2d_set_transformation_none(map2d: Map2D):
    map2d.transformation = None
    assert map2d.transformation is None
