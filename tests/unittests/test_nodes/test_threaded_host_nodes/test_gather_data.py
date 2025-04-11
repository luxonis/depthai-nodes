from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import GatheredData
from depthai_nodes.node.gather_data import GatherData

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
    return PipelineMock().create(GatherData)


def create_nn_data(timestamp=None):
    nn_data = dai.NNData()
    tensor = np.random.rand(1, 10).astype(np.float32)
    nn_data.addTensor(name="test_layer", tensor=tensor)
    if timestamp:
        nn_data.setTimestamp(timestamp)
    return nn_data


@pytest.fixture
def nn_data_in_tolerance(in_tolerance_timestamp):
    return create_nn_data(timestamp=in_tolerance_timestamp)


@pytest.fixture
def nn_data_out_of_tolerance(out_of_tolerance_timestamp):
    return create_nn_data(timestamp=out_of_tolerance_timestamp)


def create_img_detections(timestamp=None, num_detections=2):
    det = dai.ImgDetections()
    det.detections = [dai.ImgDetection() for _ in range(num_detections)]
    for i, d in enumerate(det.detections):
        d.xmin = 0.3
        d.xmax = 0.5
        d.ymin = 0.3
        d.ymax = 0.5
        d.label = i
        d.confidence = 0.9
    if timestamp:
        det.setTimestamp(timestamp)
    return det


@pytest.fixture
def img_detections(reference_timestamp):
    return create_img_detections(timestamp=reference_timestamp)


@pytest.fixture
def old_img_detections(old_reference_timestamp):
    return create_img_detections(timestamp=old_reference_timestamp)


def setup_gather_data_node(generator, fps, duration=None):
    gather_data = generator.build(camera_fps=fps)
    if duration is not None:
        gather_data.input_data._queue.duration = duration
        gather_data.input_reference._queue.duration = duration
    return gather_data


def send_data_and_run(gather_data, reference_data, data_items):
    gather_data.input_reference.send(reference_data)
    for item in data_items:
        gather_data.input_data.send(item)
    output = gather_data.out.createOutputQueue()
    gather_data.run()
    return output.getAll()


def assert_gather_data_result_basics(
    result, expected_reference, expected_gathered_count
):
    assert isinstance(result, GatheredData)
    assert result.reference_data == expected_reference
    assert len(result.gathered) == expected_gathered_count
    assert result.getTimestamp() == expected_reference.getTimestamp()


def assert_timestamps_in_tolerance(timestamps, reference_timestamp, tolerance):
    tolerance_td = timedelta(seconds=tolerance)
    for timestamp in timestamps:
        assert (
            reference_timestamp - tolerance_td
            <= timestamp
            <= reference_timestamp + tolerance_td
        )


def assert_results_not_empty(results):
    assert isinstance(results, list)
    assert len(results) > 0


def test_build(gather_data_generator, fps):
    with pytest.raises(ValueError):
        gather_data_generator.build(camera_fps=-fps)

    gather_data = gather_data_generator.build(camera_fps=fps)
    assert gather_data is not None


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
    duration,
):
    gather_data = setup_gather_data_node(
        generator=gather_data_generator, fps=fps, duration=duration
    )

    data_items = [nn_data_in_tolerance, nn_data_in_tolerance, nn_data_out_of_tolerance]
    results = send_data_and_run(
        gather_data=gather_data, reference_data=img_detections, data_items=data_items
    )

    assert_results_not_empty(results=results)

    for result in results:
        assert_gather_data_result_basics(
            result=result, expected_reference=img_detections, expected_gathered_count=2
        )

        gathered_timestamps = [gathered.getTimestamp() for gathered in result.gathered]
        assert_timestamps_in_tolerance(
            timestamps=gathered_timestamps,
            reference_timestamp=reference_timestamp,
            tolerance=tolerance,
        )

        for gathered in result.gathered:
            assert isinstance(gathered, dai.Buffer)


def test_set_wait_count_fn(gather_data_generator, fps, duration, nn_data_in_tolerance):
    gather_data = setup_gather_data_node(
        generator=gather_data_generator, fps=fps, duration=duration
    )

    gather_data.set_wait_count_fn(lambda _: 1)
    results = send_data_and_run(
        gather_data=gather_data,
        reference_data=nn_data_in_tolerance,
        data_items=[nn_data_in_tolerance],
    )

    assert_results_not_empty(results=results)

    for result in results:
        assert_gather_data_result_basics(
            result=result,
            expected_reference=nn_data_in_tolerance,
            expected_gathered_count=1,
        )
        assert result.gathered[0] == nn_data_in_tolerance


def test_clear_old_data(
    gather_data_generator,
    fps,
    duration,
    old_img_detections,
    img_detections,
    nn_data_in_tolerance,
):
    gather_data = setup_gather_data_node(
        generator=gather_data_generator, fps=fps, duration=duration
    )

    gather_data.input_reference.send(old_img_detections)
    gather_data.input_reference.send(img_detections)

    data_items = [nn_data_in_tolerance, nn_data_in_tolerance]
    results = send_data_and_run(
        gather_data=gather_data, reference_data=img_detections, data_items=data_items
    )

    assert_results_not_empty(results=results)

    for result in results:
        assert_gather_data_result_basics(
            result=result, expected_reference=img_detections, expected_gathered_count=2
        )
