import dataclasses
from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    BoxDimensions,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
)
from tests.utils.messages.creators.box import (
    BOX_DIMENSIONS,
    create_box_measurement_result,
    create_box_measurements,
    create_rotated_rect,
)

TRACKED = dai.Tracklet.TrackingStatus.TRACKED


def result_fields(**overrides) -> dict:
    return {
        "track_id": 7,
        "bounding_box": create_rotated_rect(),
        "confidence": 0.8,
        "track_status": TRACKED,
        "measurement_status": MeasurementStatus.FAILED,
        **overrides,
    }


def test_dimensions_have_a_volume():
    dimensions = BoxDimensions(length_mm=200, width_mm=100, height_mm=50)

    assert dimensions.volume_mm3 == 1_000_000


def test_dimensions_are_immutable_keyword_only_built_in_floats():
    dimensions = BoxDimensions(
        length_mm=np.int64(200),  # pyright: ignore[reportArgumentType]
        width_mm=np.float32(100),  # pyright: ignore[reportArgumentType]
        height_mm=50,
    )

    assert all(
        value.__class__ is float
        for value in (dimensions.length_mm, dimensions.width_mm, dimensions.height_mm)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        dimensions.height_mm = 60  # type: ignore[misc]
    with pytest.raises(TypeError):
        BoxDimensions(200, 100, 50)  # type: ignore[misc]


@pytest.mark.parametrize("value", ["50", True, None])
def test_dimensions_reject_wrong_types(value):
    with pytest.raises(TypeError):
        BoxDimensions(length_mm=200, width_mm=100, height_mm=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_dimensions_reject_non_finite_values(value):
    with pytest.raises(ValueError):
        BoxDimensions(length_mm=200, width_mm=100, height_mm=value)


def test_measurement_status_values_are_strings():
    assert MeasurementStatus.INELIGIBLE == "ineligible"
    assert MeasurementStatus.FAILED == "failed"
    assert MeasurementStatus.MEASURED == "measured"


def test_result_has_direct_fields_and_is_immutable():
    result = create_box_measurement_result()

    assert result.track_id == 7
    assert result.bounding_box.angle == 0
    assert result.dimensions == BOX_DIMENSIONS
    assert create_box_measurements([result]).items == [result]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.track_id = 8  # type: ignore[misc]


def test_result_without_dimensions_defaults_to_none():
    assert BoxMeasurementResult(**result_fields()).dimensions is None


def test_measured_result_requires_dimensions():
    with pytest.raises(ValueError):
        BoxMeasurementResult(
            **result_fields(measurement_status=MeasurementStatus.MEASURED)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("track_id", 7.5),
        ("bounding_box", (0.5, 0.5)),
        ("confidence", "0.8"),
        ("track_status", "tracked"),
        ("measurement_status", "failed"),
        ("dimensions", (200, 100, 50)),
    ],
)
def test_result_rejects_wrong_types(field, value):
    with pytest.raises(TypeError):
        BoxMeasurementResult(**result_fields(**{field: value}))


def test_constructor_takes_no_timestamps():
    with pytest.raises(TypeError):
        BoxMeasurements([], timestamp=timedelta(seconds=2))  # type: ignore[call-arg]


def annotation_texts(message: dai.ImgAnnotations) -> list[str]:
    return [text.text for ann in message.annotations for text in ann.texts]


def outlines(message: dai.ImgAnnotations) -> list[list[tuple[float, float]]]:
    return [
        [(point.x, point.y) for point in points.points]
        for ann in message.annotations
        for points in ann.points
    ]


def test_measured_box_displays_confidence_and_aggregated_units():
    message = create_box_measurements(
        [create_box_measurement_result()]
    ).getVisualizationMessage()

    assert annotation_texts(message) == [
        "Box 7 (80%)\nH: 5.0 cm\nL: 20.0 cm W: 10.0 cm\nV: 1000 cm³"
    ]


def test_box_without_dimensions_displays_identity_and_confidence_only():
    message = create_box_measurements(
        [create_box_measurement_result(MeasurementStatus.FAILED, None)]
    ).getVisualizationMessage()

    assert annotation_texts(message) == ["Box 7 (80%)"]


def test_box_is_outlined_by_its_bounding_box():
    message = create_box_measurements(
        [create_box_measurement_result()]
    ).getVisualizationMessage()

    drawn = outlines(message)
    assert len(drawn) == 1
    assert np.allclose(drawn[0], [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])


def test_empty_measurements_render_nothing():
    message = create_box_measurements().getVisualizationMessage()

    assert annotation_texts(message) == []
    assert outlines(message) == []


def test_annotations_carry_source_metadata():
    measurements = create_box_measurements([create_box_measurement_result()])
    measurements.setTimestamp(timedelta(seconds=2))
    measurements.setTimestampDevice(timedelta(seconds=12))
    measurements.setSequenceNum(9)

    message = measurements.getVisualizationMessage()

    assert message.getTimestamp() == timedelta(seconds=2)
    assert message.getTimestampDevice() == timedelta(seconds=12)
    assert message.getSequenceNum() == 9
