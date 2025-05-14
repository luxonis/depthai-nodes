import time
from typing import Union

import depthai as dai
import numpy as np
import pytest
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node.depth_merger import DepthMerger
from depthai_nodes.node.host_spatials_calc import HostSpatialsCalc
from tests.utils import (
    LOG_INTERVAL,
    OutputMock,
    create_img_detection,
    create_img_detection_extended,
    create_img_detections,
    create_img_detections_extended,
)

from .utils.calibration_handler import get_calibration_handler

DIFFERENT_TESTS = 4  # used for stability tests


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


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
    return create_img_detection()


@pytest.fixture
def img_detection_extended():
    return create_img_detection_extended()


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
    return create_img_detections()


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended()


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


def test_initialization(depth_merger: DepthMerger):
    assert depth_merger.shrinking_factor == 0.0


@pytest.mark.parametrize("detection", ["img_detection", "img_detection_extended"])
def test_img_detection(
    duration: float,
    depth_merger: DepthMerger,
    depth_frame: dai.ImgFrame,
    request: FixtureRequest,
    detection: str,
):
    img_detection: Union[
        ImgDetectionExtended, dai.ImgDetection
    ] = request.getfixturevalue(detection)
    output_2d = OutputMock()
    output_depth = OutputMock()

    modified_duration = duration / DIFFERENT_TESTS

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

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < modified_duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time()-start_time:.1f}s elapsed, {modified_duration-time.time()+start_time:.1f}s remaining"
            )
            last_log_time = time.time()
        output_2d.send(img_detection)
        output_depth.send(depth_frame)

        depth_merger.process(q_2d.get(), q_depth.get())

        spatial_det: dai.SpatialImgDetection = q_out.get()
        verify_spatial_detection(spatial_det, img_detection)


@pytest.mark.parametrize("detections", ["img_detections", "img_detections_extended"])
def test_img_detections(
    depth_merger: DepthMerger,
    depth_frame: dai.ImgFrame,
    request: FixtureRequest,
    detections: str,
    duration: float,
):
    img_detections: Union[
        ImgDetectionsExtended, dai.ImgDetections
    ] = request.getfixturevalue(detections)
    output_2d = OutputMock()
    output_depth = OutputMock()

    modified_duration = duration / DIFFERENT_TESTS

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

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < modified_duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time()-start_time:.1f}s elapsed, {modified_duration-time.time()+start_time:.1f}s remaining"
            )
            last_log_time = time.time()
        output_2d.send(img_detections)
        output_depth.send(depth_frame)

        depth_merger.process(q_2d.get(), q_depth.get())

        spatial_dets: dai.SpatialImgDetections = q_out.get()
        assert isinstance(spatial_dets, dai.SpatialImgDetections)
        assert len(spatial_dets.detections) == len(img_detections.detections)

        for i, spatial_det in enumerate(spatial_dets.detections):
            img_det = img_detections.detections[i]
            verify_spatial_detection(spatial_det, img_det)
