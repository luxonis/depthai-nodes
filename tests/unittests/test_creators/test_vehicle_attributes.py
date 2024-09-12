import re

import pytest

from depthai_nodes.ml.messages import VehicleAttributes
from depthai_nodes.ml.messages.creators.misc import create_vehicle_attributes_message


def test_wrong_vehicle_type():
    with pytest.raises(
        ValueError, match="Vehicle_types should be list, got <class 'str'>."
    ):
        create_vehicle_attributes_message("red", [0.3, 0.5])


def test_subtype_vehicle_type():
    with pytest.raises(
        ValueError,
        match="Vehicle_types list values must be of type float, instead got <class 'str'>.",
    ):
        create_vehicle_attributes_message(["red", 0.5], [0.3, 0.5])


def test_negative_vehicle_type():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vehicle_types list must contain probabilities between 0 and 1, instead got [-0.3, 1.5]."
        ),
    ):
        create_vehicle_attributes_message([-0.3, 1.5], ["red", 0.5])


def test_sum_vehicle_type():
    with pytest.raises(
        ValueError,
        match="Vehicle_types list must contain probabilities that sum to 1, instead got values that sum to 0.8.",
    ):
        create_vehicle_attributes_message([0.3, 0.5], ["red", 0.5])


def test_length_is_four_vehicle_type():
    with pytest.raises(
        ValueError, match=re.escape("Vehicle_types list should have 4 values, got 2.")
    ):
        create_vehicle_attributes_message([0.5, 0.5], ["red", 0.5])


def test_wrong_vehicle_color():
    with pytest.raises(
        ValueError, match="Vehicle_colors should be list, got <class 'str'>."
    ):
        create_vehicle_attributes_message([0.3, 0.5, 0.1, 0.1], "red")


def test_subtype_vehicle_color():
    with pytest.raises(
        ValueError,
        match="Vehicle_colors list values must be of type float, instead got <class 'str'>.",
    ):
        create_vehicle_attributes_message([0.3, 0.5, 0.1, 0.1], ["red", 0.5])


def test_negative_vehicle_color():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vehicle_colors list must contain probabilities between 0 and 1, instead got [-0.3, 1.5]."
        ),
    ):
        create_vehicle_attributes_message([0.3, 0.5, 0.1, 0.1], [-0.3, 1.5])


def test_sum_vehicle_color():
    with pytest.raises(
        ValueError,
        match="Vehicle_colors list must contain probabilities that sum to 1, instead got values that sum to 0.8.",
    ):
        create_vehicle_attributes_message([0.3, 0.5, 0.1, 0.1], [0.3, 0.5])


def test_length_is_seven_vehicle_color():
    with pytest.raises(
        ValueError, match="Vehicle_colors list should have 7 values, got 3."
    ):
        create_vehicle_attributes_message([0.3, 0.5, 0.1, 0.1], [0.1, 0.4, 0.5])


def test_vehicle_attributes():
    vehicle_attributes_message = create_vehicle_attributes_message(
        [0.3, 0.5, 0.1, 0.1], [0.1, 0.3, 0.1, 0.05, 0.05, 0.2, 0.2]
    )

    assert isinstance(vehicle_attributes_message, VehicleAttributes)
    assert vehicle_attributes_message.vehicle_type == ("Bus", 0.5)
    assert vehicle_attributes_message.vehicle_color == ("Gray", 0.3)


if __name__ == "__main__":
    pytest.main()
