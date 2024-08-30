from typing import List, Union

import numpy as np

from ...messages import Classifications


def create_classification_message(
    classes: List, scores: Union[np.ndarray, List]
) -> Classifications:
    """Create a message for classification. The message contains the class names and
    their respective scores, sorted in descending order of scores.

    @param classes: A list containing class names.
    @type classes: List
    @param scores: A numpy array of shape (n_classes,) containing the probability score of each class.
    @type scores: np.ndarray

    @return: A message with attributes `classes` and `scores`. `classes` is a list of classes, sorted in descending order of scores. `scores` is a list of the corresponding scores.
    @rtype: Classifications

    @raises ValueError: If the provided classes are None.
    @raises ValueError: If the provided classes are not a list.
    @raises ValueError: If the provided classes are empty.
    @raises ValueError: If the provided scores are None.
    @raises ValueError: If the provided scores are not a list or a numpy array.
    @raises ValueError: If the provided scores are empty.
    @raises ValueError: If the provided scores are not a 1D array.
    @raises ValueError: If the provided scores are not of type float.
    @raises ValueError: If the provided scores do not sum to 1.
    @raises ValueError: If the number of labels and scores mismatch.
    """
    if isinstance(classes, type(None)):
        raise ValueError("Classes should not be None.")

    if not isinstance(classes, list):
        raise ValueError(f"Classes should be a list, got {type(classes)}.")

    if len(classes) == 0:
        raise ValueError("Classes should not be empty.")

    if type(scores) == type(None):
        raise ValueError("Scores should not be None.")

    if not isinstance(scores, np.ndarray) and not isinstance(scores, list):
        raise ValueError(
            f"Scores should be a list or a numpy array, got {type(scores)}."
        )

    if isinstance(scores, list):
        scores = np.array(scores)

    if len(scores) == 0:
        raise ValueError("Scores should not be empty.")

    if len(scores) != len(scores.flatten()):
        raise ValueError(f"Scores should be a 1D array, got {scores.shape}.")

    scores = scores.flatten()

    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(f"Scores should be of type float, got {scores.dtype}.")

    if not np.isclose(np.sum(scores), 1.0, atol=1e-2):
        raise ValueError(f"Scores should sum to 1, got {np.sum(scores)}.")

    if len(scores) != len(classes):
        raise ValueError(
            f"Number of labels and scores mismatch. Provided {len(scores)} scores and {len(classes)} class names."
        )

    classification_msg = Classifications()
    sorted_args = np.argsort(-scores, kind="stable")
    scores = scores[sorted_args]

    classification_msg.classes = [classes[i] for i in sorted_args]
    classification_msg.scores = scores.tolist()

    return classification_msg
