from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

# from depthai_nodes import
from depthai_nodes.message import (
    GatheredData,
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.node.gather_data import (
    GatherData,
)

from .conftest import PipelineMock

# Need to add because it uses PipelineMock and ThreadedHostNodeMock from stability_tests conftest.py
dai.Pipeline = PipelineMock

NUMBER_OF_MESSAGES_TESTED = 0


@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def fps():
    return 30


@pytest.fixture
def tolerance(fps):
    return 1 / GatherData.FPS_TOLERANCE_DIVISOR / fps


@pytest.fixture
def reference_timestamp():
    return timedelta(days=1, hours=1, minutes=1, seconds=1, milliseconds=0)


@pytest.fixture
def in_tolerance_timestamp(reference_timestamp, tolerance):
    return reference_timestamp + timedelta(seconds=(tolerance * 0.9))


@pytest.fixture
def out_of_tolerance_timestamp(reference_timestamp, tolerance):
    return reference_timestamp + timedelta(seconds=(tolerance * 1.1))


@pytest.fixture
def gather_data_generator():
    pipeline = PipelineMock()
    return pipeline.create(GatherData)


@pytest.fixture
def nn_data():
    nn_data = dai.NNData()
    tensor = np.random.rand(1, 10).astype(np.float32)
    nn_data.addTensor("test_layer", tensor)
    return nn_data


@pytest.fixture
def nn_data_in_tolerance(nn_data, in_tolerance_timestamp):
    nn_data.setTimestamp(in_tolerance_timestamp)
    return nn_data


@pytest.fixture
def nn_data_out_of_tolerance(nn_data, out_of_tolerance_timestamp):
    nn_data.setTimestamp(out_of_tolerance_timestamp)
    return nn_data


@pytest.fixture
def img_detections(reference_timestamp):
    det = dai.ImgDetections()
    det.detections = [dai.ImgDetection() for _ in range(2)]
    for i, d in enumerate(det.detections):
        d.xmin = 0.3
        d.xmax = 0.5
        d.ymin = 0.3
        d.ymax = 0.5
        d.label = i
        d.confidence = 0.9
    det.setTimestamp(reference_timestamp)
    return det


@pytest.fixture
def img_detections_extended(reference_timestamp):
    det = ImgDetectionsExtended()
    det.detections = [ImgDetectionExtended() for _ in range(2)]
    for i, d in enumerate(det.detections):
        xmin = 0.3
        xmax = 0.5
        ymin = 0.3
        ymax = 0.5
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        d.rotated_rect = (x_center, y_center, width, height, 0)
        d.rotated_rect.angle = 0
        d.label = i
        d.confidence = 0.9
    det.setTimestamp(reference_timestamp)
    return det


def test_build(
    gather_data_generator,
):
    with pytest.raises(ValueError):
        gather_data_generator.build(camera_fps=-1)
    gather_data_generator.build(camera_fps=15)


def test_img_detections(
    gather_data_generator,
    img_detections,
    nn_data_in_tolerance,
    nn_data_out_of_tolerance,
    duration: int,
):
    gather_data: GatherData = gather_data_generator.build(camera_fps=15)
    if duration is not None:
        gather_data.input_data._queue.duration = duration
        gather_data.input_reference._queue.duration = duration

    gather_data.input_reference.send(img_detections)
    gather_data.input_data.send(nn_data_in_tolerance)
    gather_data.input_data.send(nn_data_in_tolerance)
    gather_data.input_data.send(nn_data_out_of_tolerance)

    output = gather_data.out.createOutputQueue()

    gather_data.run()
    results = output.getAll()

    assert isinstance(results, list)
    assert len(results) > 0, "The node did not send out any messages."
    for result in results:
        assert isinstance(
            result, GatheredData
        ), "All results should be GatheredData objects"
        assert (
            result.reference_data == img_detections
        ), "The reference data should match the sent ImgDetections"
        assert (
            len(result.gathered) == 2
        ), "The number of gathered data should match the number of sent nn_data messages with timestamps in tolerance"


def test_two_stage_sync_node_img_detections_extended(
    gather_data_generator,
    img_detections_extended,
    nn_data,
    duration: int,
):
    two_stage_sync = gather_data_generator.build(camera_fps=15)

    # Send data
    if duration is not None:
        two_stage_sync.input_detections._queue.duration = duration
        two_stage_sync.input_recognitions._queue.duration = duration
    two_stage_sync.input_detections.send(img_detections_extended)
    two_stage_sync._detections[
        img_detections_extended.getTimestamp().total_seconds()
    ] = img_detections_extended

    # Send two nn_data messages because we have two detections
    two_stage_sync.input_recognitions.send(nn_data)
    two_stage_sync.input_recognitions.send(nn_data)
    two_stage_sync.out.createOutputQueue(model_slug=None, parser_name=None)

    # Process the data
    two_stage_sync.run()

    global NUMBER_OF_MESSAGES_TESTED
    assert NUMBER_OF_MESSAGES_TESTED > 0, "The node did not send out any messages."
