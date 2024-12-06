import numpy as np
import pytest

from depthai_nodes.ml.messages import Map2D
from depthai_nodes.ml.messages.creators.map import create_map_message


def test_valid_2d_input():
    map_array = np.random.rand(480, 640).astype(np.float32)
    message = create_map_message(map_array)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, map_array)


def test_valid_3d_input_nhw():
    map_array = np.random.rand(1, 480, 640).astype(np.float32)
    message = create_map_message(map_array)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, map_array[0])


def test_valid_3d_input_hwn():
    map_array = np.random.rand(480, 640, 1).astype(np.float32)
    message = create_map_message(map_array)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, map_array[:, :, 0])


def test_min_max_scaling():
    map_array = np.random.rand(480, 640).astype(np.float32) * 100
    message = create_map_message(map_array, min_max_scaling=True)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.all(message.map >= 0) and np.all(message.map <= 1)
    assert np.allclose(message.map, map_array / 100, atol=1e-3)


def test_invalid_type():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'list'>."):
        create_map_message([[0.1, 0.2], [0.3, 0.4]])


def test_invalid_shape():
    with pytest.raises(ValueError, match="Expected 2D or 3D input, got 1D input."):
        create_map_message(np.array([0.1, 0.2, 0.3, 0.4]))


def test_invalid_3d_shape():
    with pytest.raises(
        ValueError, match="Unexpected map shape. Expected NHW or HWN, got"
    ):
        create_map_message(np.random.rand(2, 480, 640).astype(np.float32))


def test_valid_input_non_float():
    map_array = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    message = create_map_message(map_array)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, map_array, atol=1e-3)
