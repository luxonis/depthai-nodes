import time
from typing import List, Union

import depthai as dai
import numpy as np
import pytest
from conftest import Output, PipelineMock

# from depthai_nodes import 
from depthai_nodes.message import DetectedRecognitions, ImgDetectionExtended, ImgDetectionsExtended


@pytest.fixture
def detections_recognitions_sync():
    """Create a DetectionsRecognitionsSync instance for testing."""
    from depthai_nodes.node.detections_recognitions_sync import DetectionsRecognitionsSync

    pipeline = PipelineMock()
    return pipeline.create(DetectionsRecognitionsSync)


@pytest.fixture
def img_detection():
    det = dai.ImgDetection()
    det.xmin = 0.3
    det.xmax = 0.5
    det.ymin = 0.3
    det.ymax = 0.5
    det.label = 1
    det.confidence = 0.9
    return det


@pytest.fixture
def img_detection_extended():
    det = ImgDetectionExtended()
    xmin = 0.3
    xmax = 0.5
    ymin = 0.3
    ymax = 0.5
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    det.rotated_rect = (x_center, y_center, width, height, 0)
    det.rotated_rect.angle = 0
    det.label = 1
    det.confidence = 0.9
    return det


@pytest.fixture
def nn_data():
    """Create a single NNData object for testing."""
    nn_data = dai.NNData()
    tensor = np.random.rand(1, 10).astype(np.float32)
    nn_data.addTensor("test_layer", tensor)
    return nn_data


@pytest.fixture
def nn_data_list():
    """Create a list of NNData objects with the same timestamp."""
    data_list = []
    for i in range(2):
        nn_data = dai.NNData()
        tensor = np.random.rand(1, 10).astype(np.float32)
        nn_data.addTensor(f"test_layer_{i}", tensor)
        data_list.append(nn_data)
    return data_list


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
    return det


def test_initialization(detections_recognitions_sync):
    """Test that the sync node initializes with expected default values."""
    assert len(detections_recognitions_sync._unmatched_recognitions) == 0
    assert len(detections_recognitions_sync._recognitions_by_detection_ts) == 0
    assert len(detections_recognitions_sync._detections) == 0


def test_set_camera_fps(detections_recognitions_sync):
    """Test setting the camera FPS."""
    detections_recognitions_sync.set_camera_fps(60)
    assert detections_recognitions_sync._camera_fps == 60


def test_timestamps_in_tolerance(detections_recognitions_sync):
    """Test the timestamp tolerance function."""
    # Set camera FPS to 30, the tolerance should be 1/(30*2.0) = 0.01667 seconds
    detections_recognitions_sync.set_camera_fps(30)
    
    # Timestamps that should be within tolerance
    assert detections_recognitions_sync._timestamps_in_tolerance(1.0, 1.01)
    assert detections_recognitions_sync._timestamps_in_tolerance(1.0, 0.99)
    
    # Timestamps that should be outside tolerance
    assert not detections_recognitions_sync._timestamps_in_tolerance(1.0, 1.02)
    assert not detections_recognitions_sync._timestamps_in_tolerance(1.0, 0.98)
    
    # Change FPS to 60, tolerance becomes smaller: 1/(60*2.0) = 0.00833 seconds
    detections_recognitions_sync.set_camera_fps(60)
    
    # Now these should be outside tolerance with higher FPS
    assert not detections_recognitions_sync._timestamps_in_tolerance(1.0, 1.01)
    assert not detections_recognitions_sync._timestamps_in_tolerance(1.0, 0.99)


# def test_synchronization_with_matching_timestamps(detections_recognitions_sync, img_detections, nn_data):
#     """Test synchronization with perfectly matching timestamps."""
#     # Initialize and build
#     detections_recognitions_sync.build()
    
#     # Create output queue
#     output_queue = detections_recognitions_sync.output.createOutputQueue()
    
#     # timestamp = dai.Timestamp()
#     # timestamp.sec = 1000
#     # timestamp.nsec = 0
#     # nn_data.setTimestamp(timestamp)
    
#     # Send data
#     detections_recognitions_sync.input_detections.send(img_detections)
#     detections_recognitions_sync.input_recognitions.send(nn_data)
    
#     # Run once to process
#     detections_recognitions_sync._add_detection(img_detections)
#     detections_recognitions_sync._add_recognition(nn_data)
#     detections_recognitions_sync._send_ready_data()
    
#     # Get result
#     result = output_queue.get()
    
#     # Check result
#     assert isinstance(result, DetectedRecognitions)
#     assert result.img_detections == img_detections
#     assert len(result.nn_data) == 1
#     assert result.nn_data[0] == nn_data
