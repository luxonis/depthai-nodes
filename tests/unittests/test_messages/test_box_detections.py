import dataclasses
from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import BoxDetection, BoxDetections
from tests.utils.messages.creators.box import (
    create_box_detection,
    create_box_detections,
    create_rotated_rect,
)

TRACKED = dai.Tracklet.TrackingStatus.TRACKED


@pytest.fixture
def detections() -> BoxDetections:
    mask = np.full((4, 6), 255, dtype=np.uint8)
    mask[0, 0:2] = 0
    mask[3, 4:6] = 1
    return create_box_detections(
        [create_box_detection(7), create_box_detection(9, (0.8, 0.8))], mask
    )


def test_box_detection_stores_numbers_as_built_ins():
    detection = BoxDetection(
        track_id=np.int64(7),  # pyright: ignore[reportArgumentType]
        bounding_box=create_rotated_rect(),
        confidence=np.float32(0.5),  # pyright: ignore[reportArgumentType]
        track_status=TRACKED,
    )

    assert detection.track_id.__class__ is int
    assert detection.track_id == 7
    assert detection.confidence.__class__ is float
    assert detection.confidence == 0.5


def test_box_detection_is_immutable_and_keyword_only():
    with pytest.raises(dataclasses.FrozenInstanceError):
        create_box_detection().track_id = 8  # type: ignore[misc]
    with pytest.raises(TypeError):
        BoxDetection(7, create_rotated_rect(), 0.8, TRACKED)  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("track_id", 7.0),
        ("track_id", True),
        ("track_id", "7"),
        ("bounding_box", (0.5, 0.5)),
        ("confidence", "0.8"),
        ("confidence", True),
        ("track_status", 1),
    ],
)
def test_box_detection_rejects_wrong_types(field, value):
    fields = {
        "track_id": 7,
        "bounding_box": create_rotated_rect(),
        "confidence": 0.8,
        "track_status": TRACKED,
    }
    with pytest.raises(TypeError):
        BoxDetection(**{**fields, field: value})


def test_box_detection_rejects_non_finite_confidence():
    with pytest.raises(ValueError):
        BoxDetection(
            track_id=7,
            bounding_box=create_rotated_rect(),
            confidence=float("nan"),
            track_status=TRACKED,
        )


def test_item_mask_selects_only_pixels_labelled_with_its_index(detections):
    first = detections.getMask(0)
    second = detections.getMask(1)

    assert first.dtype == np.bool_
    assert first.shape == (4, 6)
    assert list(zip(*np.nonzero(first))) == [(0, 0), (0, 1)]
    assert list(zip(*np.nonzero(second))) == [(3, 4), (3, 5)]


def test_background_belongs_to_no_item(detections):
    background = ~(detections.getMask(0) | detections.getMask(1))

    assert int(background.sum()) == 4 * 6 - 4


@pytest.mark.parametrize("index", [-1, 2, 255])
def test_index_outside_the_items_is_rejected(detections, index):
    with pytest.raises(IndexError):
        detections.getMask(index)


def test_mask_labels_must_index_items_or_be_background():
    mask = np.full((4, 6), 255, dtype=np.uint8)
    mask[0, 0] = 1

    with pytest.raises(ValueError):
        BoxDetections([create_box_detection()], mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.full((4, 6, 1), 255, dtype=np.uint8),
        np.full((4, 6), 255, dtype=np.int32),
    ],
)
def test_mask_must_be_a_two_dimensional_label_image(mask):
    with pytest.raises(ValueError):
        BoxDetections([], mask)


def test_mask_that_is_not_an_array_is_a_type_error():
    with pytest.raises(TypeError):
        BoxDetections([], [[255]])  # type: ignore[arg-type]


def test_constructor_takes_no_timestamps():
    with pytest.raises(TypeError):
        BoxDetections(
            [],
            np.full((4, 6), 255, dtype=np.uint8),
            timestamp=timedelta(seconds=2),  # type: ignore[call-arg]
        )


def many_detections(count: int) -> list[BoxDetection]:
    return [create_box_detection(track_id) for track_id in range(count)]


def test_up_to_255_detections_fit():
    message = create_box_detections(many_detections(254))

    message.append(create_box_detection())

    assert len(message.items) == 255


def test_more_than_255_detections_are_rejected_at_construction():
    with pytest.raises(ValueError):
        create_box_detections(many_detections(256))


def test_assigning_more_than_255_detections_keeps_the_old_items(detections):
    old_items = detections.items

    with pytest.raises(ValueError):
        detections.items = many_detections(256)
    assert detections.items is old_items


def test_append_beyond_255_detections_is_rejected():
    message = create_box_detections(many_detections(255))

    with pytest.raises(ValueError):
        message.append(create_box_detection())
    assert len(message.items) == 255


def test_extend_beyond_255_detections_is_rejected(detections):
    with pytest.raises(ValueError):
        detections.extend(many_detections(254))
    assert len(detections.items) == 2

    detections.extend(many_detections(253))
    assert len(detections.items) == 255


def test_empty_detections_carry_an_all_background_mask():
    message = create_box_detections()

    assert message.items == []
    assert message.segmentation_mask.shape == (4, 6)
    assert np.all(message.segmentation_mask == 255)


def annotation_texts(message: dai.ImgAnnotations) -> list[str]:
    return [text.text for ann in message.annotations for text in ann.texts]


def outlines(message: dai.ImgAnnotations) -> list[list[tuple[float, float]]]:
    return [
        [(point.x, point.y) for point in points.points]
        for ann in message.annotations
        for points in ann.points
    ]


def test_each_box_is_labelled_with_track_and_confidence():
    message = create_box_detections(
        [
            create_box_detection(7),
            create_box_detection(12, confidence=0.55),
        ]
    ).getVisualizationMessage()

    assert annotation_texts(message) == ["Box 7 (80%)", "Box 12 (55%)"]


def test_each_box_is_outlined_by_its_bounding_box():
    message = create_box_detections(
        [create_box_detection(7), create_box_detection(12)]
    ).getVisualizationMessage()

    drawn = outlines(message)
    assert len(drawn) == 2
    assert np.allclose(drawn[0], [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])


def test_empty_detections_render_nothing():
    message = create_box_detections().getVisualizationMessage()

    assert annotation_texts(message) == []
    assert outlines(message) == []


def test_annotations_carry_source_metadata():
    detections = create_box_detections([create_box_detection()])
    detections.setTimestamp(timedelta(seconds=2))
    detections.setTimestampDevice(timedelta(seconds=12))
    detections.setSequenceNum(9)

    message = detections.getVisualizationMessage()

    assert message.getTimestamp() == timedelta(seconds=2)
    assert message.getTimestampDevice() == timedelta(seconds=12)
    assert message.getSequenceNum() == 9
