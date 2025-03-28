from typing import Union

import depthai as dai
import pytest
from conftest import Output
from pytest import FixtureRequest
from utils.create_message import create_img_detections, create_img_detections_extended

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsBridge

DETS = [
    {"bbox": [0.00, 0.00, 0.25, 0.25], "label": 0, "confidence": 0.25},
    {"bbox": [0.25, 0.25, 0.50, 0.50], "label": 1, "confidence": 0.50},
    {"bbox": [0.50, 0.50, 0.75, 0.75], "label": 2, "confidence": 0.75},
    {"bbox": [0.75, 0.75, 1.00, 1.00], "label": 3, "confidence": 1.00},
]


@pytest.fixture
def img_detections():
    return create_img_detections(DETS)


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended(DETS)


def test_initialization():
    ImgDetectionsBridge()


def test_building():
    ImgDetectionsBridge().build(Output())


@pytest.mark.parametrize(
    "img_detections_type",
    ["img_detections", "img_detections_extended"],
)
def test_processing(
    request: FixtureRequest,
    img_detections_type: str,
):
    dets: Union[ImgDetectionsExtended, dai.ImgDetections] = request.getfixturevalue(
        img_detections_type
    )

    o_dets = Output()
    bridge = ImgDetectionsBridge().build(o_dets)
    q_dets = o_dets.createOutputQueue()
    q_dets_transformed = bridge.out.createOutputQueue()

    q_dets.send(dets)
    bridge.process(q_dets.get())
    dets_transformed = q_dets_transformed.get()

    def _identical_detections(
        img_dets: dai.ImgDetections, img_dets_ext: ImgDetectionExtended
    ):
        assert isinstance(img_dets, dai.ImgDetections)
        assert isinstance(img_dets_ext, ImgDetectionsExtended)
        assert len(img_dets.detections) == len(img_dets_ext.detections)
        for img_det, img_det_ext in zip(img_dets.detections, img_dets_ext.detections):
            xmin, ymin, xmax, ymax = img_det_ext.rotated_rect.getOuterRect()
            assert img_det.xmin == xmin
            assert img_det.ymin == ymin
            assert img_det.xmax == xmax
            assert img_det.ymax == ymax
            assert img_det.label == img_det_ext.label
            assert img_det.confidence == img_det_ext.confidence

    if isinstance(dets, dai.ImgDetections):
        _identical_detections(dets, dets_transformed)
    elif isinstance(dets, ImgDetectionsExtended):
        _identical_detections(dets_transformed, dets)
    else:
        raise TypeError(f"Unexpected output message type: {type(dets)}")
