import numpy as np
import pytest

from depthai_nodes import Map2D
from depthai_nodes.message.creators import create_map_message

MAP_ARRAY = np.random.rand(1, 480, 640).astype(np.float32)


def test_valid_2d_input():
    message = create_map_message(MAP_ARRAY[0])

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, MAP_ARRAY[0])


def test_valid_3d_input_nhw():
    message = create_map_message(MAP_ARRAY)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, MAP_ARRAY[0])


def test_valid_3d_input_hwn():
    message = create_map_message(MAP_ARRAY.transpose(1, 2, 0))

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, MAP_ARRAY[0])


def test_min_max_scaling():
    map_array = MAP_ARRAY[0] * 100
    message = create_map_message(map_array, min_max_scaling=True)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.all(message.map >= 0) and np.all(message.map <= 1)
    assert np.allclose(message.map, MAP_ARRAY[0], atol=1e-3)


def test_invalid_type():
    with pytest.raises(ValueError):
        create_map_message(MAP_ARRAY.tolist())


def test_invalid_shape():
    with pytest.raises(ValueError):
        create_map_message(np.array([0.1, 0.2, 0.3, 0.4]))


def test_invalid_3d_shape():
    with pytest.raises(ValueError):
        create_map_message(np.random.rand(2, 480, 640).astype(np.float32))


def test_valid_input_non_float():
    map_array = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    message = create_map_message(map_array)

    assert isinstance(message, Map2D)
    assert message.map.shape == (480, 640)
    assert message.map.dtype == np.float32
    assert np.allclose(message.map, map_array, atol=1e-3)
