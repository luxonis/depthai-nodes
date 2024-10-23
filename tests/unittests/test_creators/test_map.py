import re

import numpy as np
import pytest

from depthai_nodes.ml.messages import Map2D
from depthai_nodes.ml.messages.creators.map import create_map_message

np.random.seed(0)


def test_not_numpy_array():
    with pytest.raises(ValueError, match="Expected numpy array, got <class 'list'>."):
        create_map_message([1, 2, 3])


def test_not_2d_or_3d_input():
    with pytest.raises(ValueError, match="Expected 2D or 3D input, got 1D input."):
        create_map_message(np.array([1, 2, 3]))


def test_wrong_input_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Unexpected map shape. Expected NHW or HWN, got (3, 1, 3)."),
    ):
        create_map_message(np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]))


def test_map_with_scaling():
    map = np.random.rand(320, 640, 1)
    map[0, 0, 0] = 0.0  # ensure that min==0
    map[1, 0, 0] = 1.0  # ensure that max==1
    unscaled_map = map * 10

    message = create_map_message(map=unscaled_map, min_max_scaling=True)

    assert isinstance(message, Map2D)
    assert message.width == 640
    assert message.height == 320

    scaled_map = message.map
    assert scaled_map.shape == map[:, :, 0].shape
    assert np.all(np.isclose(map[:, :, 0], scaled_map))


def test_map_without_scaling():
    map = np.random.rand(320, 640, 1)
    unscaled_map = map * 10

    message = create_map_message(map=unscaled_map, min_max_scaling=False)

    assert isinstance(message, Map2D)
    assert message.width == 640
    assert message.height == 320

    unscaled_map2 = message.map
    assert unscaled_map2.shape == unscaled_map[:, :, 0].shape
    assert np.all(np.isclose(unscaled_map[:, :, 0], unscaled_map2))


if __name__ == "__main__":
    pytest.main()
