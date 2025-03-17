import time

import depthai as dai
import numpy as np
import pytest
from conftest import Output

from depthai_nodes.node import TilesPatcher, Tiling


@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def dummy_tile_manager():
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


def create_dummy_nn_output(detection):
    nn_output = dai.ImgDetections()
    nn_output.detections = [detection]
    return nn_output


@pytest.fixture
def dummy_detection_1():
    det = dai.ImgDetection()
    det.label = 1
    det.confidence = 0.9
    det.xmin = 0.1
    det.ymin = 0.1
    det.xmax = 0.2
    det.ymax = 0.2
    return det


@pytest.fixture
def dummy_detection_2():
    det = dai.ImgDetection()
    det.label = 2
    det.confidence = 0.8
    det.xmin = 0.7
    det.ymin = 0.7
    det.xmax = 0.8
    det.ymax = 0.8
    return det


@pytest.fixture
def dummy_output():
    return Output()


def test_build_valid(dummy_tile_manager, dummy_output):
    patcher = TilesPatcher()
    patcher.build(
        tile_manager=dummy_tile_manager,
        nn=dummy_output,
        conf_thresh=0.5,
        iou_thresh=0.6,
    )

    assert patcher.conf_thresh == 0.5
    assert patcher.iou_thresh == 0.6
    assert patcher.tile_manager is dummy_tile_manager
    assert patcher.expected_tiles_count == len(dummy_tile_manager.tile_positions)


def test_process_accumulation(
    dummy_tile_manager, dummy_detection_1, dummy_detection_2, dummy_output, duration
):
    patcher = TilesPatcher()

    patcher.build(tile_manager=dummy_tile_manager, nn=dummy_output)

    nn_output1 = create_dummy_nn_output(dummy_detection_1)
    nn_output2 = create_dummy_nn_output(dummy_detection_2)

    out_q = patcher.out.createOutputQueue()

    patcher.process(nn_output1)
    assert out_q.is_empty()
    patcher.process(nn_output2)
    assert len(out_q._messages) == 1
    detections_msg = out_q.get()
    assert out_q.is_empty()
    assert isinstance(detections_msg, dai.ImgDetections)
    assert len(detections_msg.detections) == 2

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            patcher.process(nn_output1)
            patcher.process(nn_output2)
            assert len(out_q._messages) == 1
            detections_msg = out_q.get()
            assert out_q.is_empty()
            assert isinstance(detections_msg, dai.ImgDetections)
            assert len(detections_msg.detections) == 2
