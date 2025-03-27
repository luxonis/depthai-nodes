from datetime import timedelta
from typing import Optional

import depthai as dai
import numpy as np
import pytest
from stability_tests.conftest import PipelineMock

# from depthai_nodes import
from depthai_nodes.message import (
    DetectedRecognitions,
    ImgDetectionExtended,
    ImgDetectionsExtended,
)

NUMBER_OF_MESSAGES_TESTED = 0


@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def detections_recognitions_sync_generator():
    """Create a DetectionsRecognitionsSync instance for testing."""
    from depthai_nodes.node.detections_recognitions_sync import (
        DetectionsRecognitionsSync,
    )

    pipeline = PipelineMock()
    return pipeline.create(DetectionsRecognitionsSync)


@pytest.fixture
def nn_data():
    """Create a NNData object with the same timestamp."""
    nn_data = dai.NNData()
    tensor = np.random.rand(1, 10).astype(np.float32)
    nn_data.addTensor("test_layer", tensor)
    nn_data.setTimestamp(
        timedelta(days=1, hours=1, minutes=1, seconds=1, milliseconds=5)
    )
    return nn_data


@pytest.fixture
def img_detections():
    """Create ImgDetections with two detection objects."""
    det = dai.ImgDetections()
    det.detections = [dai.ImgDetection() for _ in range(2)]
    for i, d in enumerate(det.detections):
        d.xmin = 0.3
        d.xmax = 0.5
        d.ymin = 0.3
        d.ymax = 0.5
        d.label = i
        d.confidence = 0.9

    det.setTimestamp(timedelta(days=1, hours=1, minutes=1, seconds=1, milliseconds=0))
    return det


@pytest.fixture
def img_detections_extended():
    """Create ImgDetectionsExtended with two detection objects."""
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

    det.setTimestamp(timedelta(days=1, hours=1, minutes=1, seconds=1, milliseconds=0))
    return det


def check_synchronized_detections_recognitions(
    item, model_slug: Optional[str], parser_name: Optional[str]
) -> bool:
    """Check that detections and recognitions are properly synchronized.

    Parameters
    ----------
    item : DetectedRecognitions
        The output from DetectionsRecognitionsSync
    model_slug : Optional[str]
        Model identifier (not used in this function but required by interface)
    parser_name : Optional[str]
        Parser name (not used in this function but required by interface)

    Returns
    -------
    bool
        True if validation passes

    Raises
    ------
    AssertionError
        If the synchronization validation fails
    """
    # Constants from the DetectionsRecognitionsSync class
    FPS_TOLERANCE_DIVISOR = 2.0
    CAMERA_FPS = 30  # Default value from the DetectionsRecognitionsSync class
    global NUMBER_OF_MESSAGES_TESTED
    NUMBER_OF_MESSAGES_TESTED += 1

    # Check that the item is a DetectedRecognitions object
    assert isinstance(
        item, DetectedRecognitions
    ), f"Expected DetectedRecognitions, got {type(item)}"

    # Check that detections and nn_data are present
    assert item.img_detections is not None, "img_detections is None"
    assert item.nn_data is not None, "nn_data is None"
    assert len(item.nn_data) > 0, "nn_data is empty"

    # Get detection timestamp
    detection_ts = item.img_detections.getTimestamp().total_seconds()

    # Calculate tolerance using the same formula as in the DetectionsRecognitionsSync class
    tolerance = 1 / (CAMERA_FPS * FPS_TOLERANCE_DIVISOR)

    # Check that each recognition has a timestamp within tolerance of the detection timestamp
    for i, recognition in enumerate(item.nn_data):
        rec_ts = recognition.getTimestamp().total_seconds()
        timestamp_diff = abs(detection_ts - rec_ts)
        assert timestamp_diff < tolerance, (
            f"Timestamp difference ({timestamp_diff}) exceeds tolerance ({tolerance}): "
            f"detection={detection_ts}, recognition[{i}]={rec_ts}"
        )

    # Check that the number of recognitions matches the number of detections
    # (if there are detections)
    if (
        hasattr(item.img_detections, "detections")
        and len(item.img_detections.detections) > 0
    ):
        assert len(item.nn_data) == len(item.img_detections.detections), (
            f"Number of recognitions ({len(item.nn_data)}) doesn't match "
            f"number of detections ({len(item.img_detections.detections)})"
        )

    # For ImgDetectionsExtended, perform additional checks
    if isinstance(item.img_detections, ImgDetectionsExtended):
        for detection in item.img_detections.detections:
            assert isinstance(
                detection, ImgDetectionExtended
            ), f"Expected ImgDetectionExtended but got {type(detection)}"

    return True


def test_detections_recognitions_sync_node_build_valid(
    detections_recognitions_sync_generator,
):
    detections_recognitions_sync = detections_recognitions_sync_generator.build()

    assert len(detections_recognitions_sync._unmatched_recognitions) == 0
    assert len(detections_recognitions_sync._recognitions_by_detection_ts) == 0
    assert len(detections_recognitions_sync._detections) == 0


def test_detections_recognitions_sync_node_img_detections(
    detections_recognitions_sync_generator, img_detections, nn_data, duration: int
):
    detections_recognitions_sync = detections_recognitions_sync_generator.build()

    # Send data
    if duration is not None:
        detections_recognitions_sync.input_detections._queue.duration = duration
        detections_recognitions_sync.input_recognitions._queue.duration = duration
    detections_recognitions_sync.input_detections.send(img_detections)
    detections_recognitions_sync._detections[
        img_detections.getTimestamp().total_seconds()
    ] = img_detections

    # Send two nn_data messages because we have two detections
    detections_recognitions_sync.input_recognitions.send(nn_data)
    detections_recognitions_sync.input_recognitions.send(nn_data)
    detections_recognitions_sync.out.createOutputQueue(
        check_synchronized_detections_recognitions, model_slug=None, parser_name=None
    )

    # Process the data
    detections_recognitions_sync.run()

    global NUMBER_OF_MESSAGES_TESTED
    assert NUMBER_OF_MESSAGES_TESTED > 0, "The node did not send out any messages."


def test_detections_recognitions_sync_node_img_detections_extended(
    detections_recognitions_sync_generator,
    img_detections_extended,
    nn_data,
    duration: int,
):
    detections_recognitions_sync = detections_recognitions_sync_generator.build()

    # Send data
    if duration is not None:
        detections_recognitions_sync.input_detections._queue.duration = duration
        detections_recognitions_sync.input_recognitions._queue.duration = duration
    detections_recognitions_sync.input_detections.send(img_detections_extended)
    detections_recognitions_sync._detections[
        img_detections_extended.getTimestamp().total_seconds()
    ] = img_detections_extended

    # Send two nn_data messages because we have two detections
    detections_recognitions_sync.input_recognitions.send(nn_data)
    detections_recognitions_sync.input_recognitions.send(nn_data)
    detections_recognitions_sync.out.createOutputQueue(
        check_synchronized_detections_recognitions, model_slug=None, parser_name=None
    )

    # Process the data
    detections_recognitions_sync.run()

    global NUMBER_OF_MESSAGES_TESTED
    assert NUMBER_OF_MESSAGES_TESTED > 0, "The node did not send out any messages."
