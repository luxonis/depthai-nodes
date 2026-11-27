from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import BoxDetections
from depthai_nodes.node.box_detector.box_detection_assembler import (
    BORDER_MARGIN,
    MAX_BBOX_AREA,
    BoxDetectionAssembler,
)
from tests.utils import OutputMock

TRACKED = dai.Tracklet.TrackingStatus.TRACKED
REMOVED = dai.Tracklet.TrackingStatus.REMOVED
LOST = dai.Tracklet.TrackingStatus.LOST
NEW = dai.Tracklet.TrackingStatus.NEW
BOX_A = (0.1, 0.1, 0.3, 0.3)
BOX_B = (0.6, 0.6, 0.9, 0.9)
BOX_EMPTY = (0.4, 0.0, 0.5, 0.1)
# Overlaps BOX_A less than an exact box does, so a removed or lost track on
# BOX_A would win the pairing if it were not dropped first.
NEAR_A = (0.1, 0.1, 0.3, 0.32)
NOWHERE = (0.0, 0.8, 0.1, 0.9)
MASK_SIZE = (32, 24)


def create_tracklets(
    tracks: list[tuple[int, dai.Tracklet.TrackingStatus, tuple[float, ...]]],
) -> dai.Tracklets:
    msg = dai.Tracklets()
    tracklets = []
    for track_id, status, (xmin, ymin, xmax, ymax) in tracks:
        tracklet = dai.Tracklet()
        tracklet.id = track_id
        tracklet.status = status
        tracklet.roi = dai.Rect(dai.Point2f(xmin, ymin), dai.Point2f(xmax, ymax))
        tracklets.append(tracklet)
    msg.tracklets = tracklets
    return msg


def create_detections(
    detections: list[tuple[tuple[float, ...], float]], mask: np.ndarray | None
) -> dai.ImgDetections:
    msg = dai.ImgDetections()
    items = []
    for (xmin, ymin, xmax, ymax), confidence in detections:
        det = dai.ImgDetection()
        det.xmin, det.ymin, det.xmax, det.ymax = xmin, ymin, xmax, ymax
        det.confidence = confidence
        items.append(det)
    msg.detections = items
    if mask is not None:
        msg.setCvSegmentationMask(mask)
    msg.setTimestamp(timedelta(seconds=1))
    msg.setTimestampDevice(timedelta(seconds=11))
    msg.setSequenceNum(42)
    return msg


def background_mask() -> np.ndarray:
    return np.full((MASK_SIZE[1], MASK_SIZE[0]), 255, dtype=np.uint8)


def region(rows: slice, cols: slice) -> np.ndarray:
    expected = np.zeros((MASK_SIZE[1], MASK_SIZE[0]), dtype=bool)
    expected[rows, cols] = True
    return expected


def assemble(tracklets: dai.Tracklets, detections: dai.ImgDetections):
    assembler = BoxDetectionAssembler().build(OutputMock(), OutputMock(), MASK_SIZE)
    queue = assembler.out.createOutputQueue()
    assembler.process(tracklets, detections)
    message = queue.get()
    assert isinstance(message, BoxDetections)
    return message


def test_tracked_box_becomes_a_box_detection_with_the_frame_metadata():
    mask = background_mask()
    mask[3:17, 7:25] = 0
    message = assemble(
        create_tracklets([(7, TRACKED, (0.2, 0.2, 0.8, 0.8))]),
        create_detections([((0.2, 0.2, 0.8, 0.8), 0.8)], mask),
    )

    assert message.getTimestamp() == timedelta(seconds=1)
    assert message.getTimestampDevice() == timedelta(seconds=11)
    assert message.getSequenceNum() == 42
    (item,) = message.items
    assert item.track_id == 7
    assert item.track_status == TRACKED
    assert item.confidence == pytest.approx(0.8)
    assert item.bounding_box.center.x == pytest.approx(0.5)
    assert item.bounding_box.center.y == pytest.approx(0.5)
    assert item.bounding_box.size.width == pytest.approx(0.6)
    assert item.bounding_box.size.height == pytest.approx(0.6)
    assert item.bounding_box.angle == 0
    np.testing.assert_array_equal(
        message.getMask(0), region(slice(3, 17), slice(7, 25))
    )


def test_only_current_tracks_with_a_nonempty_segmentation_are_kept():
    mask = background_mask()
    mask[2:7, 3:10] = 0
    mask[14:22, 19:29] = 1
    message = assemble(
        create_tracklets(
            [
                (3, NEW, BOX_B),
                (4, REMOVED, BOX_A),
                (6, LOST, BOX_A),
                (5, TRACKED, NEAR_A),
                (9, TRACKED, NOWHERE),
                (11, TRACKED, BOX_EMPTY),
            ]
        ),
        create_detections([(BOX_A, 0.9), (BOX_B, 0.7), (BOX_EMPTY, 0.6)], mask),
    )

    by_track = {item.track_id: (i, item) for i, item in enumerate(message.items)}
    assert sorted(by_track) == [3, 5]
    index_b, item_b = by_track[3]
    index_a, item_a = by_track[5]
    assert item_b.track_status == NEW
    assert item_b.confidence == pytest.approx(0.7)
    assert item_b.bounding_box.center.x == pytest.approx(0.75)
    assert item_a.bounding_box.size.width == pytest.approx(0.2)
    np.testing.assert_array_equal(
        message.getMask(index_a), region(slice(2, 7), slice(3, 10))
    )
    np.testing.assert_array_equal(
        message.getMask(index_b), region(slice(14, 22), slice(19, 29))
    )
    assert set(np.unique(message.segmentation_mask).tolist()) == {0, 1, 255}


@pytest.mark.parametrize(("area_offset", "emitted"), [(-1e-3, True), (1e-3, False)])
def test_detections_covering_more_than_the_max_area_are_dropped(area_offset, emitted):
    width = 0.8
    bounds = (0.1, 0.1, 0.1 + width, 0.1 + (MAX_BBOX_AREA + area_offset) / width)
    mask = background_mask()
    mask[3:17, 7:25] = 0
    message = assemble(
        create_tracklets([(7, TRACKED, bounds)]),
        create_detections([(bounds, 0.8)], mask),
    )

    assert [item.track_id for item in message.items] == ([7] if emitted else [])


def margin_pixels(size: int) -> int:
    """Pixels on each side whose centre lies within BORDER_MARGIN of the edge."""
    return int(BORDER_MARGIN * size + 0.5)


def edge_probe(edge: str, inset: int) -> tuple[slice, slice]:
    """Rows and columns of a block reaching to ``inset`` pixels from one edge."""
    width, height = MASK_SIZE
    rows, cols = slice(8, 16), slice(10, 22)
    if edge == "left":
        cols = slice(inset, cols.stop)
    elif edge == "right":
        cols = slice(cols.start, width - inset)
    elif edge == "top":
        rows = slice(inset, rows.stop)
    else:
        rows = slice(rows.start, height - inset)
    return rows, cols


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize(("extra_inset", "emitted"), [(-1, False), (0, True)])
def test_truncated_detections_are_dropped_without_taking_a_mask_index(
    edge, extra_inset, emitted
):
    size = MASK_SIZE[0] if edge in ("left", "right") else MASK_SIZE[1]
    probe = edge_probe(edge, margin_pixels(size) + extra_inset)
    inside = (slice(3, 6), slice(3, 6))
    mask = background_mask()
    mask[probe] = 0
    mask[inside] = 1
    message = assemble(
        create_tracklets([(4, TRACKED, BOX_B), (5, TRACKED, BOX_A)]),
        create_detections([(BOX_B, 0.9), (BOX_A, 0.8)], mask),
    )

    assert [item.track_id for item in message.items] == ([4, 5] if emitted else [5])
    np.testing.assert_array_equal(
        message.getMask(len(message.items) - 1), region(*inside)
    )
    if not emitted:
        assert set(np.unique(message.segmentation_mask).tolist()) == {0, 255}


def test_bounding_box_in_the_margin_with_its_mask_inside_is_emitted():
    bounds = (0.0, 0.0, 0.6, 0.6)
    mask = background_mask()
    mask[3:12, 3:16] = 0
    message = assemble(
        create_tracklets([(7, TRACKED, bounds)]),
        create_detections([(bounds, 0.8)], mask),
    )

    assert [item.track_id for item in message.items] == [7]


def test_truncated_box_keeps_its_track_id_once_inside():
    assembler = BoxDetectionAssembler().build(OutputMock(), OutputMock(), MASK_SIZE)
    queue = assembler.out.createOutputQueue()
    entering = background_mask()
    entering[edge_probe("left", 0)] = 0
    inside = background_mask()
    inside[3:17, 7:25] = 0

    for mask in (entering, inside):
        assembler.process(
            create_tracklets([(7, TRACKED, BOX_A)]),
            create_detections([(BOX_A, 0.8)], mask),
        )
    truncated, entered = queue.get(), queue.get()

    assert isinstance(truncated, BoxDetections)
    assert isinstance(entered, BoxDetections)
    assert truncated.items == []
    (item,) = entered.items
    assert item.track_id == 7
    assert item.track_status == TRACKED


def test_frame_without_a_mask_emits_empty_detections_of_the_mask_size():
    message = assemble(create_tracklets([]), create_detections([], None))

    assert message.items == []
    assert message.getSequenceNum() == 42
    assert message.segmentation_mask.shape == (MASK_SIZE[1], MASK_SIZE[0])
    assert (message.segmentation_mask == 255).all()
