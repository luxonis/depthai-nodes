from typing import List

from ...messages import AgeGender


def create_age_gender_message(age: float, gender_prob: List[float]) -> AgeGender:
    """Create a DepthAI message for the age and gender probability.

    @param age: Detected person age (must be multiplied by 100 to get years).
    @type age: float
    @param gender_prob: Detected person gender probability [female, male].
    @type gender_prob: List[float]
    @return: AgeGender message containing the detected person age and gender
        probability.
    @rtype: AgeGender
    @raise ValueError: If age is not a float.
    @raise ValueError: If gender_prob is not a list.
    @raise ValueError: If each item in gender_prob is not a float.
    """

    if not isinstance(age, float):
        raise ValueError(f"age should be float, got {type(age)}.")

    if not isinstance(gender_prob, List):
        raise ValueError(f"gender_prob should be list, got {type(gender_prob)}.")
    for item in gender_prob:
        if not isinstance(item, float):
            raise ValueError(
                f"gender_prob list values must be of type float, instead got {type(item)}."
            )

    age_gender_message = AgeGender()
    age_gender_message.age = age
    age_gender_message.gender_prob = gender_prob

    return age_gender_message
