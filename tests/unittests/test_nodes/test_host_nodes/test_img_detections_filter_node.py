from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import ImgDetectionsFilter
from tests.utils import OutputMock


def create_img_detections(
    detections_spec: list[tuple[int, float, tuple[float, float, float, float]]],
    mask: np.ndarray | None = None,
) -> dai.ImgDetections:
    msg = dai.ImgDetections()
    msg.setTimestamp(timedelta(seconds=1))
    msg.setSequenceNum(11)
    detections = []

    for label, confidence, (xmin, ymin, xmax, ymax) in detections_spec:
        det = dai.ImgDetection()
        det.label = label
        det.confidence = confidence
        det.xmin = xmin
        det.ymin = ymin
        det.xmax = xmax
        det.ymax = ymax
        detections.append(det)

    msg.detections = detections

    if mask is not None:
        msg.setCvSegmentationMask(mask)

    return msg


@pytest.fixture
def filter_node() -> ImgDetectionsFilter:
    return ImgDetectionsFilter()


def process_message(
    filter_node: ImgDetectionsFilter, msg: dai.ImgDetections
) -> dai.ImgDetections:
    input_mock = OutputMock()
    filter_node.build(input_mock)
    output_queue = filter_node.out.createOutputQueue()
    filter_node.process(msg)
    output = output_queue.get()
    assert isinstance(output, dai.ImgDetections)
    return output


def test_keeps_only_requested_labels(filter_node: ImgDetectionsFilter):
    msg = create_img_detections(
        [
            (1, 0.90, (0.10, 0.10, 0.20, 0.20)),
            (2, 0.80, (0.20, 0.20, 0.30, 0.30)),
            (1, 0.70, (0.30, 0.30, 0.40, 0.40)),
        ]
    )

    output = process_message(filter_node.keepLabels([1]), msg)

    assert [det.label for det in output.detections] == [1, 1]
    assert [det.confidence for det in output.detections] == pytest.approx([0.90, 0.70])


def test_rejects_labels_and_updates_segmentation_mask(
    filter_node: ImgDetectionsFilter,
):
    msg = create_img_detections(
        [
            (1, 0.90, (0.10, 0.10, 0.20, 0.20)),
            (2, 0.80, (0.20, 0.20, 0.35, 0.35)),
            (3, 0.70, (0.30, 0.30, 0.45, 0.45)),
        ],
        mask=np.array(
            [
                [0, 1, 2],
                [2, 1, 0],
            ],
            dtype=np.uint8,
        ),
    )

    output = process_message(filter_node.rejectLabels([2]), msg)

    assert [det.label for det in output.detections] == [1, 3]
    np.testing.assert_array_equal(
        output.getCvSegmentationMask(),
        np.array(
            [
                [0, 255, 2],
                [2, 255, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_filters_by_confidence_and_area(filter_node: ImgDetectionsFilter):
    msg = create_img_detections(
        [
            (1, 0.95, (0.10, 0.10, 0.50, 0.50)),
            (2, 0.60, (0.10, 0.10, 0.20, 0.20)),
            (3, 0.85, (0.10, 0.10, 0.15, 0.15)),
        ]
    )

    output = process_message(filter_node.minConfidence(0.80).minArea(0.05), msg)

    assert len(output.detections) == 1
    assert output.detections[0].label == 1
    assert output.detections[0].confidence == pytest.approx(0.95)


def test_sorts_by_confidence_before_taking_top_k(filter_node: ImgDetectionsFilter):
    msg = create_img_detections(
        [
            (1, 0.40, (0.10, 0.10, 0.30, 0.30)),
            (2, 0.95, (0.20, 0.20, 0.40, 0.40)),
            (3, 0.70, (0.30, 0.30, 0.50, 0.50)),
        ]
    )

    output = process_message(
        filter_node.sortByConfidence(desc=True).takeFirstK(2),
        msg,
    )

    assert [det.label for det in output.detections] == [2, 3]
    assert [det.confidence for det in output.detections] == pytest.approx([0.95, 0.70])


def test_disable_sorting_keeps_original_order_before_top_k(
    filter_node: ImgDetectionsFilter,
):
    msg = create_img_detections(
        [
            (1, 0.40, (0.10, 0.10, 0.30, 0.30)),
            (2, 0.95, (0.20, 0.20, 0.40, 0.40)),
            (3, 0.70, (0.30, 0.30, 0.50, 0.50)),
        ]
    )

    output = process_message(
        filter_node.sortByConfidence(desc=True).disableSorting().takeFirstK(2),
        msg,
    )

    assert [det.label for det in output.detections] == [1, 2]


def test_nms_suppresses_overlapping_detections_of_same_label(
    filter_node: ImgDetectionsFilter,
):
    msg = create_img_detections(
        [
            (1, 0.95, (0.10, 0.10, 0.50, 0.50)),
            (1, 0.80, (0.12, 0.12, 0.48, 0.48)),
            (1, 0.70, (0.60, 0.60, 0.80, 0.80)),
        ]
    )

    output = process_message(
        filter_node.useNms(confThresh=0.0, iouThresh=0.5),
        msg,
    )

    assert len(output.detections) == 2
    assert [det.confidence for det in output.detections] == pytest.approx([0.95, 0.70])


def test_nms_keeps_overlapping_detections_of_different_labels(
    filter_node: ImgDetectionsFilter,
):
    msg = create_img_detections(
        [
            (1, 0.95, (0.10, 0.10, 0.50, 0.50)),
            (2, 0.80, (0.12, 0.12, 0.48, 0.48)),
        ]
    )

    output = process_message(
        filter_node.useNms(confThresh=0.0, iouThresh=0.5),
        msg,
    )

    assert [det.label for det in output.detections] == [1, 2]
