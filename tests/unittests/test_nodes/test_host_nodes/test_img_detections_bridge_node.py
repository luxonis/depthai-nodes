import math
import time
from typing import Union

import depthai as dai
import pytest
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsBridge
from tests.utils import (
    LOG_INTERVAL,
    OutputMock,
    create_img_detections,
    create_img_detections_extended,
)

img_det_types = ["img_detections", "img_detections_extended"]


@pytest.fixture(scope="session")
def duration(request):
    d = request.config.getoption("--duration")
    if d is None:
        return 1e-6
    return d


@pytest.fixture
def bridge():
    return ImgDetectionsBridge()


@pytest.fixture
def img_detections():
    return create_img_detections()


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended()


def test_building(bridge: ImgDetectionsBridge):
    bridge.build(OutputMock())


@pytest.mark.parametrize(
    "img_detections_type",
    img_det_types,
)
def test_processing(
    bridge: ImgDetectionsBridge,
    request: FixtureRequest,
    img_detections_type: str,
    duration: float,
):
    dets: Union[ImgDetectionsExtended, dai.ImgDetections] = request.getfixturevalue(
        img_detections_type
    )

    o_dets = OutputMock()
    bridge.build(o_dets, ignore_angle=True)
    q_dets = o_dets.createOutputQueue()
    q_dets_transformed = bridge.out.createOutputQueue()

    modified_duration = duration / len(img_det_types)

    def _identical_detections(
        img_dets: dai.ImgDetections, img_dets_ext: ImgDetectionExtended
    ):
        assert isinstance(img_dets, dai.ImgDetections)
        assert isinstance(img_dets_ext, ImgDetectionsExtended)
        assert len(img_dets.detections) == len(img_dets_ext.detections)
        for img_det, img_det_ext in zip(img_dets.detections, img_dets_ext.detections):
            xmin, ymin, xmax, ymax = img_det_ext.rotated_rect.getOuterRect()
            assert math.isclose(img_det.xmin, xmin, rel_tol=1e-6)
            assert math.isclose(img_det.ymin, ymin, rel_tol=1e-6)
            assert math.isclose(img_det.xmax, xmax, rel_tol=1e-6)
            assert math.isclose(img_det.ymax, ymax, rel_tol=1e-6)
            assert math.isclose(img_det.label, img_det_ext.label, rel_tol=1e-6)
            assert math.isclose(
                img_det.confidence, img_det_ext.confidence, rel_tol=1e-6
            )

    start_time = time.time()
    last_log_time = time.time()
    while time.time() - start_time < modified_duration:
        if time.time() - last_log_time > LOG_INTERVAL:
            print(
                f"Test running... {time.time()-start_time:.1f}s elapsed, {modified_duration-time.time()+start_time:.1f}s remaining"
            )
            last_log_time = time.time()
        q_dets.send(dets)
        bridge.process(q_dets.get())
        dets_transformed = q_dets_transformed.get()

        if isinstance(dets, dai.ImgDetections):
            _identical_detections(dets, dets_transformed)
        elif isinstance(dets, ImgDetectionsExtended):
            _identical_detections(dets_transformed, dets)
        else:
            raise TypeError(f"Unexpected output message type: {type(dets)}")
