import time

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import Tiling
from tests.utils import OutputMock


@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def dummy_img_frame():
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    img_frame = dai.ImgFrame()
    img_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
    return img_frame


@pytest.fixture
def tiling():
    return Tiling()


def test_build(tiling):
    output = OutputMock()
    grid_size = (2, 2)
    img_shape = (1280, 720)
    nn_shape = np.array([300, 300])
    overlap = 0.1

    tiling.build(
        overlap=overlap,
        img_output=output,
        grid_size=grid_size,
        img_shape=img_shape,
        nn_shape=nn_shape,
        global_detection=False,
    )

    assert tiling.overlap == overlap
    assert tiling.grid_size == grid_size
    np.testing.assert_allclose(tiling.nn_shape, nn_shape)
    assert tiling.img_shape == img_shape

    n_tiles_w, n_tiles_h = grid_size
    A = np.array(
        [
            [n_tiles_w * (1 - overlap) + overlap, 0],
            [0, n_tiles_h * (1 - overlap) + overlap],
        ]
    )
    b = np.array(img_shape)
    expected_tile_dims = np.linalg.inv(A).dot(b)
    np.testing.assert_allclose(tiling.x, expected_tile_dims)

    expected_num_tiles = grid_size[0] * grid_size[1]
    assert len(tiling.tile_positions) == expected_num_tiles


def test_build_global_detection(tiling, dummy_img_frame):
    output = OutputMock()
    grid_size = (2, 2)
    img_shape = (dummy_img_frame.getWidth(), dummy_img_frame.getHeight())
    nn_shape = np.array([300, 300])
    overlap = 0.1

    tiling.build(
        overlap=overlap,
        img_output=output,
        grid_size=grid_size,
        img_shape=img_shape,
        nn_shape=nn_shape,
        global_detection=True,
    )

    first_tile = tiling.tile_positions[0]
    assert first_tile["coords"] == (0, 0, img_shape[0], img_shape[1])

    scale = min(nn_shape[0] / img_shape[0], nn_shape[1] / img_shape[1])
    expected_scaled_width = int(img_shape[0] * scale)
    expected_scaled_height = int(img_shape[1] * scale)
    assert first_tile["scaled_size"] == (expected_scaled_width, expected_scaled_height)


def test_process(tiling, dummy_img_frame, duration):
    output = OutputMock()
    grid_size = (2, 2)
    img_shape = (dummy_img_frame.getWidth(), dummy_img_frame.getHeight())
    nn_shape = np.array([300, 300])
    overlap = 0.1
    out_q = tiling.out.createOutputQueue()

    tiling.build(
        overlap=overlap,
        img_output=output,
        grid_size=grid_size,
        img_shape=img_shape,
        nn_shape=nn_shape,
        global_detection=False,
    )

    tiling.process(dummy_img_frame)

    expected_num_tiles = len(tiling.tile_positions)

    for _ in range(expected_num_tiles):
        out_frame = out_q.get()
        assert isinstance(out_frame, dai.ImgFrame)

    assert out_q.is_empty()

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            tiling.process(dummy_img_frame)
            for _ in range(expected_num_tiles):
                out_frame = out_q.get()
                assert isinstance(out_frame, dai.ImgFrame)
            assert out_q.is_empty()
