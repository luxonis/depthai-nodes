import time
from typing import Union

import depthai as dai
import pytest
from pytest import FixtureRequest

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsFilter

from tests.utils import OutputMock
from .utils.create_message import (
    DETECTIONS,
    create_img_detections,
    create_img_detections_extended,
)

LABELS = [1]
CONF_THRES = 0.5
MAX_DET = 1
SORT = True


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture
def img_detections():
    return create_img_detections()


@pytest.fixture
def img_detections_extended():
    return create_img_detections_extended()


def test_initialization():
    filter = ImgDetectionsFilter()
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject is None
    assert filter._confidence_threshold is None
    assert filter._max_detections is None
    assert filter._sort_by_confidence is False


def test_building():
    filter = ImgDetectionsFilter().build(OutputMock())
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject is None
    assert filter._confidence_threshold is None
    assert filter._max_detections is None
    assert filter._sort_by_confidence is False

    # labels to keep
    filter = ImgDetectionsFilter().build(
        OutputMock(),
        labels_to_keep=LABELS,
    )
    assert filter._labels_to_keep == LABELS
    assert filter._labels_to_reject is None

    # labels to reject
    filter = ImgDetectionsFilter().build(
        OutputMock(),
        labels_to_reject=LABELS,
    )
    assert filter._labels_to_keep is None
    assert filter._labels_to_reject == LABELS

    # labels_to_keep and labels_to_reject cannot be set at the same time
    with pytest.raises(ValueError):
        ImgDetectionsFilter().build(
            OutputMock(),
            labels_to_keep=LABELS,
            labels_to_reject=LABELS,
        )

    # confidence_threshold
    filter = ImgDetectionsFilter().build(
        OutputMock(),
        confidence_threshold=CONF_THRES,
    )
    assert filter._confidence_threshold == CONF_THRES

    # max_detections
    filter = ImgDetectionsFilter().build(
        OutputMock(),
        max_detections=MAX_DET,
    )
    assert filter._max_detections == MAX_DET

    # sort_by_confidence
    filter = ImgDetectionsFilter().build(
        OutputMock(),
        sort_by_confidence=SORT,
    )
    assert filter._sort_by_confidence == SORT


def test_parameter_setting():
    filter = ImgDetectionsFilter().build(OutputMock())

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

    # sort_by_confidence
    filter.setSortByConfidence(SORT)
    assert filter._sort_by_confidence == SORT
    with pytest.raises(ValueError):
        filter.setSortByConfidence("not a boolean")


@pytest.mark.parametrize(
    "img_detections_type",
    ["img_detections", "img_detections_extended"],
)
def test_processing(
    request: FixtureRequest,
    img_detections_type: str,
    duration: int = 1e-6,  # allows only one run
):
    dets: Union[ImgDetectionsExtended, dai.ImgDetections] = request.getfixturevalue(
        img_detections_type
    )

    o_dets = OutputMock()
    filter = ImgDetectionsFilter().build(o_dets)
    q_dets = o_dets.createOutputQueue()
    q_dets_filtered = filter.out.createOutputQueue()

    start_time = time.time()
    while time.time() - start_time < duration:
        # default filtering
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        assert isinstance(dets_filtered, type(dets))
        assert len(dets_filtered.detections) == len(DETECTIONS["bboxes"])

        # filter by labels_to_keep
        filter.setLabels(LABELS, keep=True)
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        assert len(dets_filtered.detections) == len(
            [None for label in DETECTIONS["labels"] if label in LABELS]
        )
        for det in dets_filtered.detections:
            assert det.label in LABELS
        filter.setLabels(None, keep=True)  # setting back to default

        # filter by labels_to_reject
        filter.setLabels(LABELS, keep=False)
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        assert len(dets_filtered.detections) == len(
            [None for label in DETECTIONS["labels"] if label not in LABELS]
        )
        for det in dets_filtered.detections:
            assert det.label not in LABELS
        filter.setLabels(None, keep=False)  # setting back to default

        # filter by confidence
        filter.setConfidenceThreshold(CONF_THRES)
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        assert len(dets_filtered.detections) == len(
            [None for score in DETECTIONS["scores"] if score >= CONF_THRES]
        )
        for det in dets_filtered.detections:
            assert det.confidence >= CONF_THRES
        filter.setConfidenceThreshold(None)  # setting back to default

        # filter by max detections
        filter.setMaxDetections(MAX_DET)
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        assert len(dets_filtered.detections) == MAX_DET
        filter.setMaxDetections(None)  # setting it back to default

        # sort by confidence
        filter.setSortByConfidence(SORT)
        q_dets.send(dets)
        filter.process(q_dets.get())
        dets_filtered = q_dets_filtered.get()
        if SORT:
            assert dets_filtered.detections == sorted(
                dets_filtered.detections, key=lambda x: x.confidence, reverse=True
            )
        filter.setSortByConfidence(False)
