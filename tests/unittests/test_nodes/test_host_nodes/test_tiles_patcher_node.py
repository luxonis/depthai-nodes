import time

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import TilesPatcher, Tiling
from tests.utils import OutputMock, create_img_detection


@pytest.fixture
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


@pytest.fixture
def patcher():
    return TilesPatcher()


@pytest.fixture
def tile_manager():
    tm = Tiling()
    tm.overlap = 0.1
    tm.grid_size = (2, 2)
    tm.x = np.array([640, 360])
    tm.nn_shape = (300, 300)
    tm.img_shape = (1280, 720)
    tm.tile_positions = [
        {"coords": (0, 0, 640, 360), "scaled_size": (300, 300)},
        {"coords": (640, 360, 1280, 720), "scaled_size": (300, 300)},
    ]
    return tm


@pytest.fixture
def img_detection_1():
    return create_img_detection(
        [0.1, 0.1, 0.2, 0.2],
        1,
        0.9,
    )


@pytest.fixture
def img_detection_2():
    return create_img_detection(
        [0.7, 0.7, 0.8, 0.8],
        2,
        0.8,
    )


def create_dummy_nn_output(detection):
    nn_output = dai.ImgDetections()
    nn_output.detections = [detection]
    return nn_output


def test_build_valid(patcher: TilesPatcher, tile_manager: Tiling):
    patcher.build(
        tile_manager=tile_manager,
        nn=OutputMock(),
        conf_thresh=0.5,
        iou_thresh=0.6,
    )

    assert patcher.conf_thresh == 0.5
    assert patcher.iou_thresh == 0.6
    assert patcher.tile_manager is tile_manager
    assert patcher.expected_tiles_count == len(tile_manager.tile_positions)


def test_process_accumulation(
    patcher: TilesPatcher,
    tile_manager: Tiling,
    img_detection_1,
    img_detection_2,
    duration,
):
    patcher.build(tile_manager=tile_manager, nn=OutputMock())

    nn_output1 = create_dummy_nn_output(img_detection_1)
    nn_output2 = create_dummy_nn_output(img_detection_2)

    out_q = patcher.out.createOutputQueue()

    start_time = time.time()

    while time.time() - start_time < duration:
        patcher.process(nn_output1)
        assert out_q.is_empty()
        patcher.process(nn_output2)
        assert len(out_q._messages) == 1
        detections_msg = out_q.get()
        assert out_q.is_empty()
        assert isinstance(detections_msg, dai.ImgDetections)
        assert len(detections_msg.detections) == 2
