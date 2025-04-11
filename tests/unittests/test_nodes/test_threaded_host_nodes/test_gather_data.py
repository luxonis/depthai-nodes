from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    GatheredData,
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.node.gather_data import (
    GatherData,
)

from .conftest import PipelineMock


@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def fps():
    return 30


@pytest.fixture
def tolerance(fps):
    return 1 / fps / GatherData.FPS_TOLERANCE_DIVISOR


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
def old_reference_timestamp(reference_timestamp, tolerance):
    return reference_timestamp - timedelta(seconds=(tolerance * 1.1))


@pytest.fixture
def gather_data_generator():
    pipeline = PipelineMock()
    return pipeline.create(GatherData)


def get_nn_data():
    nn_data = dai.NNData()
    tensor = np.random.rand(1, 10).astype(np.float32)
    nn_data.addTensor("test_layer", tensor)
    return nn_data


@pytest.fixture
def nn_data_in_tolerance(in_tolerance_timestamp):
    nn_data = get_nn_data()
    nn_data.setTimestamp(in_tolerance_timestamp)
    return nn_data


@pytest.fixture
def nn_data_out_of_tolerance(out_of_tolerance_timestamp):
    nn_data = get_nn_data()
    nn_data.setTimestamp(out_of_tolerance_timestamp)
    return nn_data


def get_img_detections():
    det = dai.ImgDetections()
    det.detections = [dai.ImgDetection() for _ in range(2)]
    for i, d in enumerate(det.detections):
        d.xmin = 0.3
        d.xmax = 0.5
        d.ymin = 0.3
        d.ymax = 0.5
        d.label = i
        d.confidence = 0.9
    return det


@pytest.fixture
def img_detections(reference_timestamp):
    dets = get_img_detections()
    dets.setTimestamp(reference_timestamp)
    return dets


@pytest.fixture
def old_img_detections(old_reference_timestamp):
    dets = get_img_detections()
    dets.setTimestamp(old_reference_timestamp)
    return dets


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


def test_build(gather_data_generator, fps):
    with pytest.raises(ValueError):
        gather_data_generator.build(camera_fps=-fps)
    gather_data_generator.build(camera_fps=fps)


def test_run_without_build(gather_data_generator):
    with pytest.raises(ValueError):
        gather_data_generator.run()


def test_img_detections(
    gather_data_generator,
    fps,
    img_detections,
    nn_data_in_tolerance,
    nn_data_out_of_tolerance,
    reference_timestamp,
    tolerance,
    duration: int,
):
    gather_data: GatherData = gather_data_generator.build(camera_fps=fps)
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
            result.getTimestamp() == reference_timestamp
        ), "The timestamp should match the reference timestamp"
        assert (
            result.reference_data == img_detections
        ), "The reference data should match the sent ImgDetections"
        assert (
            len(result.gathered) == 2
        ), "The number of gathered data should match the number of sent nn_data messages with timestamps in tolerance"
        for gathered in result.gathered:
            assert isinstance(
                gathered, dai.Buffer
            ), "Gathered data should be derived from dai.Buffer"
            tolerance_td = timedelta(seconds=tolerance)
            assert (
                reference_timestamp - tolerance_td
                <= gathered.getTimestamp()
                <= reference_timestamp + tolerance_td
            ), "Gathered data timestamp should be within the tolerance range"


def test_set_wait_count_fn(gather_data_generator, fps, duration, nn_data_in_tolerance):
    gather_data: GatherData[dai.NNData, dai.NNData] = gather_data_generator.build(
        camera_fps=fps
    )
    if duration is not None:
        gather_data.input_data._queue.duration = duration
        gather_data.input_reference._queue.duration = duration
    gather_data.input_reference.send(nn_data_in_tolerance)
    gather_data.input_data.send(nn_data_in_tolerance)
    gather_data.set_wait_count_fn(lambda _: 1)
    output = gather_data.out.createOutputQueue()
    gather_data.run()
    results = output.getAll()
    assert isinstance(results, list)
    assert len(results) > 0, "The node should have sent out more than one message."
    for result in results:
        assert isinstance(
            result, GatheredData
        ), "The result should be a GatheredData object"
        assert (
            result.reference_data == nn_data_in_tolerance
        ), "The reference data should match the sent ImgDetections"
        assert (
            len(result.gathered) == 1
        ), "The number of gathered data should match the wait count function"
        assert result.gathered[0] == nn_data_in_tolerance


def test_clear_old_data(
    gather_data_generator,
    fps,
    duration,
    old_img_detections,
    img_detections,
    nn_data_in_tolerance,
):
    gather_data: GatherData = gather_data_generator.build(camera_fps=fps)
    if duration is not None:
        gather_data.input_data._queue.duration = duration
        gather_data.input_reference._queue.duration = duration
    gather_data.input_reference.send(old_img_detections)
    gather_data.input_reference.send(img_detections)
    gather_data.input_data.send(nn_data_in_tolerance)
    gather_data.input_data.send(nn_data_in_tolerance)
    output = gather_data.out.createOutputQueue()
    gather_data.run()
    results = output.getAll()
    assert isinstance(results, list)
    assert len(results) > 0, "The node should have sent out more than one message."
    for result in results:
        assert isinstance(
            result, GatheredData
        ), "The result should be a GatheredData object"
        assert (
            result.reference_data == img_detections
        ), "The reference data should match the sent ImgDetections"
        assert len(result.gathered) == 2, "There should be only one gathered data item"
