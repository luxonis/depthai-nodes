import dataclasses
from datetime import timedelta

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
    create_box_view_detection,
)


def result_fields(**overrides) -> dict:
    return {
        "box_id": 7,
        "detections": (create_box_view_detection(),),
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
    detections = (create_box_view_detection(view=0), create_box_view_detection(view=1))
    result = create_box_measurement_result(
        box_id=np.int64(7),  # pyright: ignore[reportArgumentType]
        detections=detections,
    )

    assert result.box_id == 7
    assert result.box_id.__class__ is int
    assert result.detections is detections
    assert result.dimensions == BOX_DIMENSIONS
    assert create_box_measurements([result]).items == [result]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.box_id = 8  # type: ignore[misc]


def test_result_without_dimensions_defaults_to_none():
    assert BoxMeasurementResult(**result_fields()).dimensions is None


def test_measured_result_requires_dimensions():
    with pytest.raises(ValueError):
        BoxMeasurementResult(
            **result_fields(measurement_status=MeasurementStatus.MEASURED)
        )


def test_result_requires_at_least_one_detection():
    with pytest.raises(ValueError):
        BoxMeasurementResult(**result_fields(detections=()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("box_id", "7"),
        ("box_id", True),
        ("detections", [create_box_view_detection()]),
        ("detections", (create_box_view_detection(), "detection")),
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
