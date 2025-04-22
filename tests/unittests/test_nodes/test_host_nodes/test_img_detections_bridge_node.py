import math
import time
from typing import Union

import depthai as dai
import pytest
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsBridge

from .conftest import Output
from .utils.create_message import (
    create_img_detections,
    create_img_detections_extended,
)


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def bridge():
    return ImgDetectionsBridge()


@pytest.fixture
def img_detections():
    return create_img_detections()


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended()


def test_initialization(bridge: ImgDetectionsBridge):
    assert bridge.parentInitialized()


def test_building(bridge: ImgDetectionsBridge):
    bridge.build(Output())


@pytest.mark.parametrize(
    "img_detections_type",
    ["img_detections", "img_detections_extended"],
)
def test_processing(
    bridge: ImgDetectionsBridge,
    request: FixtureRequest,
    img_detections_type: str,
    duration: int = 1e-6,
):
    dets: Union[ImgDetectionsExtended, dai.ImgDetections] = request.getfixturevalue(
        img_detections_type
    )

    o_dets = Output()
    bridge.build(o_dets, ignore_angle=True)
    q_dets = o_dets.createOutputQueue()
    q_dets_transformed = bridge.out.createOutputQueue()

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
    while time.time() - start_time < duration:
        q_dets.send(dets)
        bridge.process(q_dets.get())
        dets_transformed = q_dets_transformed.get()

        if isinstance(dets, dai.ImgDetections):
            _identical_detections(dets, dets_transformed)
        elif isinstance(dets, ImgDetectionsExtended):
            _identical_detections(dets_transformed, dets)
        else:
            raise TypeError(f"Unexpected output message type: {type(dets)}")
