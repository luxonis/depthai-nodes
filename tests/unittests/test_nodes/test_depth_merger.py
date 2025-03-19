import time
from typing import Union

import depthai as dai
import numpy as np
import pytest
from conftest import Output
from pytest import FixtureRequest
from utils.calibration_handler import get_calibration_handler

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node.depth_merger import DepthMerger
from depthai_nodes.node.host_spatials_calc import HostSpatialsCalc


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def depth_merger():
    depth_merger = DepthMerger(shrinking_factor=0.0)
    calib_handler = get_calibration_handler()
    depth_merger.host_spatials_calc = HostSpatialsCalc(
        calib_data=calib_handler, depth_alignment_socket=dai.CameraBoardSocket.CAM_A
    )
    return depth_merger


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
def depth_frame():
    frame = np.random.randint(200, 30000, (720, 1280), dtype=np.uint16)
    img_frame = dai.ImgFrame()
    img_frame.setCvFrame(frame, dai.ImgFrame.Type.RAW16)
    img_frame.setWidth(1280)
    img_frame.setHeight(720)
    return img_frame


@pytest.fixture
def img_detections():
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


def verify_spatial_detection(spatial_det, img_detection):
    assert isinstance(spatial_det, dai.SpatialImgDetection)

    if isinstance(img_detection, ImgDetectionExtended):
        xmin, ymin, xmax, ymax = img_detection.rotated_rect.getOuterRect()
    else:
        xmin = img_detection.xmin
        ymin = img_detection.ymin
        xmax = img_detection.xmax
        ymax = img_detection.ymax
    np.testing.assert_almost_equal(spatial_det.xmin, xmin, decimal=2)
    np.testing.assert_almost_equal(spatial_det.ymin, ymin, decimal=2)
    np.testing.assert_almost_equal(spatial_det.xmax, xmax, decimal=2)
    np.testing.assert_almost_equal(spatial_det.ymax, ymax, decimal=2)
    assert spatial_det.label == img_detection.label
    np.testing.assert_almost_equal(
        spatial_det.confidence, img_detection.confidence, decimal=2
    )


def test_initialization(depth_merger: DepthMerger, depth_frame: dai.ImgFrame):
    assert depth_merger.shrinking_factor == 0.0


@pytest.mark.parametrize("detection", ["img_detection", "img_detection_extended"])
def test_img_detection(
    duration: int,
    depth_merger: DepthMerger,
    depth_frame: dai.ImgFrame,
    request: FixtureRequest,
    detection: str,
):
    img_detection: Union[
        ImgDetectionExtended, dai.ImgDetection
    ] = request.getfixturevalue(detection)
    output_2d = Output()
    output_depth = Output()

    depth_merger.build(
        output_2d=output_2d,
        output_depth=output_depth,
        calib_data=get_calibration_handler(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
        shrinking_factor=0.0,
    )

    q_2d = output_2d.createOutputQueue()
    q_depth = output_depth.createOutputQueue()
    q_out = depth_merger.output.createOutputQueue()

    output_2d.send(img_detection)
    output_depth.send(depth_frame)

    depth_merger.process(q_2d.get(), q_depth.get())

    spatial_det: dai.SpatialImgDetection = q_out.get()
    verify_spatial_detection(spatial_det, img_detection)

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            output_2d.send(img_detection)
            output_depth.send(depth_frame)

            depth_merger.process(q_2d.get(), q_depth.get())

            spatial_det = q_out.get()
            verify_spatial_detection(spatial_det, img_detection)


@pytest.mark.parametrize("detections", ["img_detections", "img_detections_extended"])
def test_img_detections(
    depth_merger: DepthMerger,
    depth_frame: dai.ImgFrame,
    request: FixtureRequest,
    detections: str,
    duration: int,
):
    img_detections: Union[
        ImgDetectionsExtended, dai.ImgDetections
    ] = request.getfixturevalue(detections)
    output_2d = Output()
    output_depth = Output()

    depth_merger.build(
        output_2d=output_2d,
        output_depth=output_depth,
        calib_data=get_calibration_handler(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
        shrinking_factor=0.0,
    )

    q_2d = output_2d.createOutputQueue()
    q_depth = output_depth.createOutputQueue()
    q_out = depth_merger.output.createOutputQueue()

    output_2d.send(img_detections)
    output_depth.send(depth_frame)

    depth_merger.process(q_2d.get(), q_depth.get())

    spatial_dets: dai.SpatialImgDetections = q_out.get()
    assert isinstance(spatial_dets, dai.SpatialImgDetections)
    assert len(spatial_dets.detections) == len(img_detections.detections)

    for i, spatial_det in enumerate(spatial_dets.detections):
        img_det = img_detections.detections[i]
        verify_spatial_detection(spatial_det, img_det)

    if duration:
        start_time = time.time()

        while time.time() - start_time < duration:
            output_2d.send(img_detections)
            output_depth.send(depth_frame)

            depth_merger.process(q_2d.get(), q_depth.get())

            spatial_dets = q_out.get()
            assert isinstance(spatial_dets, dai.SpatialImgDetections)
            assert len(spatial_dets.detections) == len(img_detections.detections)

            for i, spatial_det in enumerate(spatial_dets.detections):
                img_det = img_detections.detections[i]
                verify_spatial_detection(spatial_det, img_det)
