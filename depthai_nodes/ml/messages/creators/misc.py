from typing import List

from ...messages import AgeGender, Classifications, VehicleAttributes


def create_age_gender_message(age: float, gender_prob: List[float]) -> AgeGender:
    """Create a DepthAI message for the age and gender probability.

    @param age: Detected person age (must be multiplied by 100 to get years).
    @type age: float
    @param gender_prob: Detected person gender probability [female, male].
    @type gender_prob: List[float]
    @return: AgeGender message containing the predicted person's age and Classifications
        message containing the classes and probabilities of the predicted gender.
    @rtype: AgeGender
    @raise ValueError: If age is not a float.
    @raise ValueError: If gender_prob is not a list.
    @raise ValueError: If each item in gender_prob is not a float.
    """

    if not isinstance(age, float):
        raise ValueError(f"Age should be float, got {type(age)}.")

    if not isinstance(gender_prob, List):
        raise ValueError(f"Gender_prob should be list, got {type(gender_prob)}.")
    if len(gender_prob) != 2:
        raise ValueError(
            f"Gender_prob list should have 2 values, got {len(gender_prob)}."
        )

    for item in gender_prob:
        if not isinstance(item, (float)):
            raise ValueError(
                f"Gender_prob list values must be of type float, instead got {type(item)}."
            )

    if sum(gender_prob) < 0.99 or sum(gender_prob) > 1.01:
        raise ValueError(
            f"Gender_prob list must contain probabilities and sum to 1, got sum {sum(gender_prob)}."
        )

    age_gender_message = AgeGender()
    age_gender_message.age = age
    gender = Classifications()
    gender.classes = ["female", "male"]
    gender.scores = gender_prob
    age_gender_message.gender = gender

    return age_gender_message


def create_vehicle_attributes_message(
    vehicle_types: List[float], vehicle_colors: List[float]
):
    """Create a DepthAI message for the vehicle attributes.

    @param vehicle_types: Probabilities for classes [car, bus, truck, van].
    @type vehicle_types: List[float]
    @param vehicle_colors: Probabilities for classes [white, gray, yellow, red, green,
        blue, black].
    @type vehicle_colors: List[float]
    @return: VehicleAttributes message containing the predicted vehicle type and color
        with the highest probability.
    @rtype: VehicleAttributes
    @raise ValueError: If vehicle_types is not a list of floats.
    @raise ValueError: If vehicle_colors is not a list of floats.
    @raise ValueError: If vehicle_types not a probability list.
    @raise ValueError: If vehicle_colors not a probability list.
    """
    vehicle_type_classes = ["Car", "Bus", "Truck", "Van"]
    vehicle_color_classes = ["White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"]

    if not isinstance(vehicle_types, List):
        raise ValueError(f"Vehicle_types should be list, got {type(vehicle_types)}.")

    if any([not isinstance(item, float) for item in vehicle_types]):
        raise ValueError(
            f"Vehicle_types list values must be of type float, instead got {type(vehicle_types[0])}."
        )
    if any([value < 0 or value > 1 for value in vehicle_types]):
        raise ValueError(
            f"Vehicle_types list must contain probabilities between 0 and 1, instead got {vehicle_types}."
        )

    if sum(vehicle_types) < 0.99 or sum(vehicle_types) > 1.01:
        raise ValueError(
            f"Vehicle_types list must contain probabilities that sum to 1, instead got values that sum to {sum(vehicle_types)}."
        )

    if len(vehicle_types) != len(vehicle_type_classes):
        raise ValueError(
            f"Vehicle_types list should have {len(vehicle_type_classes)} values, got {len(vehicle_types)}."
        )

    if not isinstance(vehicle_colors, List):
        raise ValueError(f"Vehicle_colors should be list, got {type(vehicle_colors)}.")

    if any([not isinstance(item, float) for item in vehicle_colors]):
        raise ValueError(
            f"Vehicle_colors list values must be of type float, instead got {type(vehicle_colors[0])}."
        )

    if any([value < 0 or value > 1 for value in vehicle_colors]):
        raise ValueError(
            f"Vehicle_colors list must contain probabilities between 0 and 1, instead got {vehicle_colors}."
        )

    if sum(vehicle_colors) < 0.99 or sum(vehicle_colors) > 1.01:
        raise ValueError(
            f"Vehicle_colors list must contain probabilities that sum to 1, instead got values that sum to {sum(vehicle_colors)}."
        )

    if len(vehicle_colors) != len(vehicle_color_classes):
        raise ValueError(
            f"Vehicle_colors list should have {len(vehicle_color_classes)} values, got {len(vehicle_colors)}."
        )

    vehicle_attributes_message = VehicleAttributes()
    max_type_index = vehicle_types.index(max(vehicle_types))
    max_color_index = vehicle_colors.index(max(vehicle_colors))

    vehicle_attributes_message.vehicle_type = (
        vehicle_type_classes[max_type_index],
        round(vehicle_types[max_type_index], 4),
    )
    vehicle_attributes_message.vehicle_color = (
        vehicle_color_classes[max_color_index],
        round(vehicle_colors[max_color_index], 4),
    )

    return vehicle_attributes_message
