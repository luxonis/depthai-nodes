import time
from functools import partial
from typing import Callable
from unittest.mock import MagicMock

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import SnapsProducer, SnapsProducerFrameOnly
from tests.utils import LOG_INTERVAL, OutputMock, create_img_detection, create_img_frame

HEIGHT, WIDTH = 100, 100
FRAME = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


@pytest.fixture
def snaps_producer_frame_only():
    producer = SnapsProducerFrameOnly()
    producer._logger = MagicMock()
    return producer


@pytest.fixture
def snaps_producer():
    producer = SnapsProducer()
    producer._logger = MagicMock()
    return producer


@pytest.fixture
def simple_process_fn_frame_only():
    def filter_detections_frame_only(producer, frame):
        producer.sendSnap(name="test", frame=frame)

    return filter_detections_frame_only


@pytest.fixture
def simple_process_fn():
    def filter_detections(producer, frame, msg):
        producer.sendSnap(name="test", frame=frame)

    return filter_detections


def test_building_frame_only(snaps_producer_frame_only: SnapsProducerFrameOnly):
    snaps_producer_frame_only.build(OutputMock())


def test_building(snaps_producer: SnapsProducer):
    snaps_producer.build(OutputMock(), OutputMock())


def test_parameter_setting_frame_only(
    snaps_producer_frame_only: SnapsProducerFrameOnly,
    recwarn,
    token: str = "test_token",
    url: str = "test_url",
    time_interval: float = 1.0,
    process_fn: Callable = simple_process_fn_frame_only,
):
    # token
    snaps_producer_frame_only.setToken(token)
    assert not recwarn.list, "Unexpected warnings raised on .setToken"
    with pytest.raises(ValueError):
        snaps_producer_frame_only.setToken(1)

    # url
    snaps_producer_frame_only.setUrl(url)
    assert not recwarn.list, "Unexpected warnings raised on .setUrl"
    with pytest.raises(ValueError):
        snaps_producer_frame_only.setUrl(1)

    # time_interval
    snaps_producer_frame_only.setTimeInterval(time_interval)
    assert snaps_producer_frame_only.time_interval == time_interval
    snaps_producer_frame_only.setTimeInterval(int(time_interval))
    assert snaps_producer_frame_only.time_interval == int(time_interval)
    with pytest.raises(ValueError):
        snaps_producer_frame_only.setTimeInterval("not an integer or float")

    # process_fn
    snaps_producer_frame_only.setProcessFn(process_fn)
    with pytest.raises(ValueError):
        snaps_producer_frame_only.setProcessFn("not a function")
    with pytest.raises(ValueError):
        snaps_producer_frame_only.setProcessFn(None)


def test_parameter_setting(
    snaps_producer: SnapsProducer,
    recwarn,
    token: str = "test_token",
    url: str = "test_url",
    time_interval: float = 1.0,
    process_fn: Callable = simple_process_fn,
):
    # token
    snaps_producer.setToken(token)
    assert not recwarn.list, "Unexpected warnings raised on .setToken"
    with pytest.raises(ValueError):
        snaps_producer.setToken(1)

    # url
    snaps_producer.setUrl(url)
    assert not recwarn.list, "Unexpected warnings raised on .setUrl"
    with pytest.raises(ValueError):
        snaps_producer.setUrl(1)

    # time_interval
    snaps_producer.setTimeInterval(time_interval)
    assert snaps_producer.time_interval == time_interval
    snaps_producer.setTimeInterval(int(time_interval))
    assert snaps_producer.time_interval == int(time_interval)
    with pytest.raises(ValueError):
        snaps_producer.setTimeInterval("not an integer or float")

    # process_fn
    snaps_producer.setProcessFn(process_fn)
    with pytest.raises(ValueError):
        snaps_producer.setProcessFn("not a function")
    with pytest.raises(ValueError):
        snaps_producer.setProcessFn(None)


def test_processing_frame_only(
    snaps_producer_frame_only: SnapsProducerFrameOnly,
    duration: float,
    caplog,
):
    o_snaps = OutputMock()
    snaps_producer_frame_only.build(o_snaps, time_interval=LOG_INTERVAL)
    q_snaps = o_snaps.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

            frame = create_img_frame(FRAME)
            q_snaps.send(frame)

            with caplog.at_level("INFO"):
                snaps_producer_frame_only.process(q_snaps.get())

            snaps_producer_frame_only._logger.info.assert_called_with(
                "Snap `frame` sent"
            )


def test_processing_frame_only_custom_process_fn(
    snaps_producer_frame_only: SnapsProducerFrameOnly,
    simple_process_fn_frame_only: Callable,
    duration: float,
    caplog,
):
    o_snaps = OutputMock()
    snaps_producer_frame_only.build(
        o_snaps, time_interval=LOG_INTERVAL, process_fn=simple_process_fn_frame_only
    )
    q_snaps = o_snaps.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

            frame = create_img_frame(FRAME)
            q_snaps.send(frame)

            with caplog.at_level("INFO"):
                snaps_producer_frame_only.process(q_snaps.get())

            snaps_producer_frame_only._logger.info.assert_called_with(
                "Snap `test` sent"
            )


def test_processing(
    snaps_producer: SnapsProducer,
    duration: float,
    caplog,
):
    o_snaps_frame = OutputMock()
    o_snaps_msg = OutputMock()
    snaps_producer.build(o_snaps_frame, o_snaps_msg, time_interval=LOG_INTERVAL)
    q_snaps_frame = o_snaps_frame.createOutputQueue()
    q_snaps_msg = o_snaps_msg.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

            frame = create_img_frame(FRAME)
            q_snaps_frame.send(frame)
            detection = create_img_detection()
            q_snaps_msg.send(detection)

            with caplog.at_level("INFO"):
                snaps_producer.process(q_snaps_frame.get(), q_snaps_msg.get())

            snaps_producer._logger.info.assert_called_with("Snap `frame` sent")


def test_processing_custom_process_fn(
    snaps_producer: SnapsProducer,
    simple_process_fn: Callable,
    duration: float,
    caplog,
):
    o_snaps_frame = OutputMock()
    o_snaps_msg = OutputMock()
    snaps_producer.build(
        o_snaps_frame,
        o_snaps_msg,
        time_interval=LOG_INTERVAL,
        process_fn=simple_process_fn,
    )
    q_snaps_frame = o_snaps_frame.createOutputQueue()
    q_snaps_msg = o_snaps_msg.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

            frame = create_img_frame(FRAME)
            q_snaps_frame.send(frame)
            detection = create_img_detection()
            q_snaps_msg.send(detection)

            with caplog.at_level("INFO"):
                snaps_producer.process(q_snaps_frame.get(), q_snaps_msg.get())

            snaps_producer._logger.info.assert_called_with("Snap `test` sent")


def test_partial_process_fn(
    snaps_producer_frame_only: SnapsProducerFrameOnly,
    duration: float,
    caplog,
):
    def partial_process_fn(
        producer: SnapsProducerFrameOnly, frame: dai.ImgFrame, threshold: float
    ):
        if frame.getWidth() > threshold:
            producer.sendSnap("threshold_snap", frame)

    o_snaps = OutputMock()
    snaps_producer_frame_only.build(
        o_snaps,
        time_interval=LOG_INTERVAL,
        process_fn=partial(partial_process_fn, threshold=1),
    )
    q_snaps = o_snaps.createOutputQueue()

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time() - start_time:.1f}s elapsed, {duration - time.time() + start_time:.1f}s remaining"
            )
            last_log_time = time.time()

            frame = create_img_frame(FRAME)
            q_snaps.send(frame)

            with caplog.at_level("INFO"):
                snaps_producer_frame_only.process(q_snaps.get())

            snaps_producer_frame_only._logger.info.assert_called_with(
                "Snap `threshold_snap` sent"
            )
