from typing import List, Union

import depthai as dai
import numpy as np

from ...messages import AgeGender

def create_age_gender_message(
    age: float,
    gender_prob: List[float]
) -> AgeGender:
    """Create a message for the keypoints. The message contains 2D or 3D coordinates of
    the detected keypoints.

    Args:
        age (float): Detected person age.
        gender_prob (List[float]): Detected person gender probability [female, male].

    Returns:
        AgeGender: Message containing the detected person age and gender probability.
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