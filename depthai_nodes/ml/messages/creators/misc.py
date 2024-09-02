from typing import List

from ...messages import AgeGender, Classifications


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
