from typing import List, Union

import numpy as np

from ...messages import AgeGender, Classifications, CompositeMessage
from ...messages.creators import create_classification_message


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


def create_multi_classification_message(
    classification_attributes: List[str],
    classification_scores: Union[np.ndarray, List[List[float]]],
    classification_labels: List[List[str]],
) -> CompositeMessage:
    """Create a DepthAI message for multi-classification.

    @param classification_attributes: List of attributes being classified.
    @type classification_attributes: List[str]
    @param classification_scores: A 2D array or list of classification scores for each
        attribute.
    @type classification_scores: Union[np.ndarray, List[List[float]]]
    @param classification_labels: A 2D list of class labels for each classification
        attribute.
    @type classification_labels: List[List[str]]
    @return: MultiClassification message containing a dictionary of classification
        attributes and their respective Classifications.
    @rtype: dai.Buffer
    @raise ValueError: If number of attributes is not same as number of score-label
        pairs.
    @raise ValueError: If number of scores is not same as number of labels for each
        attribute.
    @raise ValueError: If each class score not in the range [0, 1].
    @raise ValueError: If each class score not a probability distribution that sums to
        1.
    """

    if len(classification_attributes) != len(classification_scores) or len(
        classification_attributes
    ) != len(classification_labels):
        raise ValueError(
            f"Number of classification attributes, scores and labels should be equal. Got {len(classification_attributes)} attributes, {len(classification_scores)} scores and {len(classification_labels)} labels."
        )

    multi_class_dict = {}
    for attribute, scores, labels in zip(
        classification_attributes, classification_scores, classification_labels
    ):
        if len(scores) != len(labels):
            raise ValueError(
                f"Number of scores and labels should be equal for each classification attribute, got {len(scores)} scores, {len(labels)} labels for attribute {attribute}."
            )
        multi_class_dict[attribute] = create_classification_message(labels, scores)

    multi_classification_message = CompositeMessage()
    multi_classification_message.setData(multi_class_dict)

    return multi_classification_message
