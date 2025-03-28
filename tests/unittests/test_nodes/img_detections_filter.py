from typing import Union

import depthai as dai
import pytest
from conftest import Output
from pytest import FixtureRequest
from utils.create_message import create_img_detections, create_img_detections_extended

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsFilter

DETS = [
    {"bbox": [0.00, 0.00, 0.25, 0.25], "label": 0, "confidence": 0.25},
    {"bbox": [0.25, 0.25, 0.50, 0.50], "label": 1, "confidence": 0.50},
    {"bbox": [0.50, 0.50, 0.75, 0.75], "label": 2, "confidence": 0.75},
    {"bbox": [0.75, 0.75, 1.00, 1.00], "label": 3, "confidence": 1.00},
]
LABELS = [1]
CONF_THRES = 0.5
MAX_DET = 1


@pytest.fixture
def img_detections():
    return create_img_detections(DETS)


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended(DETS)


def test_initialization():
    filter = ImgDetectionsFilter()
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject is None
    assert filter._confidence_threshold is None
    assert filter._max_detections is None


def test_building():
    filter = ImgDetectionsFilter().build(Output())
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject is None
    assert filter._confidence_threshold is None
    assert filter._max_detections is None

    # labels
    filter = ImgDetectionsFilter().build(
        Output(),
        labels_to_keep=LABELS,
    )
    assert filter._labels_to_keep == LABELS
    assert filter._labels_to_reject is None
    with pytest.raises(ValueError):
        ImgDetectionsFilter().build(
            Output(),
            labels_to_reject=LABELS,
        )
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject == LABELS
    with pytest.raises(ValueError):
        ImgDetectionsFilter().build(
            Output(),
            labels_to_keep=LABELS,
            labels_to_reject=LABELS,
        )  # labels_to_keep and labels_to_reject cannot be set at the same time

    # confidence_threshold
    filter = ImgDetectionsFilter().build(
        Output(),
        confidence_threshold=CONF_THRES,
    )
    assert filter._confidence_threshold == CONF_THRES

    # max_detections
    filter = ImgDetectionsFilter().build(
        Output(),
        max_detections=MAX_DET,
    )
    assert filter._max_detections == MAX_DET


def test_parameter_setting():
    filter = ImgDetectionsFilter().build(Output())

    # labels
    filter.setLabels(LABELS, keep=True)
    assert filter._labels_to_keep == LABELS
    filter.setLabels(LABELS, keep=False)
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject == LABELS
    with pytest.raises(ValueError):
        filter.setLabels("not a list", keep=True)
    with pytest.raises(ValueError):
        filter.setLabels(LABELS, keep="not a boolean")

    # confidence_threshold
    filter.setConfidenceThreshold(CONF_THRES)
    assert filter._confidence_threshold == CONF_THRES
    with pytest.raises(ValueError):
        filter.setConfidenceThreshold("not a float")

    # max_detections
    filter.setMaxDetections(MAX_DET)
    assert filter._max_detections == MAX_DET
    with pytest.raises(ValueError):
        filter.setMaxDetections("not an integer")


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
    filter = ImgDetectionsFilter().build(o_dets)
    q_dets = o_dets.createOutputQueue()
    q_dets_filtered = filter.out.createOutputQueue()

    # default filtering
    q_dets.send(dets)
    filter.process(q_dets.get())
    dets_filtered = q_dets_filtered.get()
    assert isinstance(dets_filtered, type(dets))
    assert len(dets_filtered.detections) == len(DETS)

    # filter by labels_to_keep
    filter.setLabels(LABELS, keep=True)
    q_dets.send(dets)
    filter.process(q_dets.get())
    dets_filtered = q_dets_filtered.get()
    assert len(dets_filtered.detections) == len(
        [det for det in DETS if det["label"] in LABELS]
    )
    for det in dets_filtered.detections:
        assert det.label in LABELS
    filter.setLabels([], keep=True)  # setting back to "default"

    # filter by labels_to_reject
    filter.setLabels(LABELS, keep=False)
    q_dets.send(dets)
    filter.process(q_dets.get())
    dets_filtered = q_dets_filtered.get()
    assert len(dets_filtered.detections) == len(
        [det for det in DETS if det["label"] not in LABELS]
    )
    for det in dets_filtered.detections:
        assert det.label not in LABELS
    filter.setLabels([], keep=False)  # setting back to "default"

    # filter by confidence
    filter.setConfidenceThreshold(CONF_THRES)
    q_dets.send(dets)
    filter.process(q_dets.get())
    dets_filtered = q_dets_filtered.get()
    assert len(dets_filtered.detections) == len(
        [det for det in DETS if det["confidence"] >= CONF_THRES]
    )
    for det in dets_filtered.detections:
        assert det.confidence >= CONF_THRES
    filter.setConfidenceThreshold(0.0)  # setting back to "default"

    # filter by max detections
    filter.setMaxDetections(MAX_DET)
    q_dets.send(dets)
    filter.process(q_dets.get())
    dets_filtered = q_dets_filtered.get()
    assert len(dets_filtered.detections) == MAX_DET
