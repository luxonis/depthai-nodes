from typing import List, Union

import numpy as np

from .. import Classifications


def create_classification_sequence_message(
    classes: List,
    scores: Union[np.ndarray, List],
    remove_duplicates: bool = False,
    ignored_indexes: List[int] = None,
) -> Classifications:
    """Creates a message for a multi-class sequence. The 'scores' array is a sequence of
    probabilities for each class at each position in the sequence. The message contains
    the class names and their respective scores, ordered according to the sequence.

    @param classes: A list of class names, with length 'n_classes'.
    @type classes: List
    @param scores: A numpy array of shape (sequence_length, n_classes) containing the (row-wise) probability distributions over the classes.
    @type scores: np.ndarray
    @param remove_duplicates: If True, removes consecutive duplicates from the sequence.
    @type remove_duplicates: bool
    @param ignored_indexes: A list of indexes to ignore during classification generation (e.g., background class, padding class)
    @type ignored_indexes: List[int]

    @return: A message with attributes `classes` and `scores`, both ordered by the sequence.
    @rtype: Classifications

    @raises ValueError: If 'classes' is not a list of strings.
    @raises ValueError: If 'scores' is not a 2D array of list of shape (sequence_length, n_classes).
    @raises ValueError: If the number of classes does not match the number of columns in 'scores'.
    @raises ValueError: If any score is not in the range [0, 1].
    @raises ValueError: If the probabilities in any row of 'scores' do not sum to 1.
    @raises ValueError: If 'ignored_indexes' in not None or a list of valid indexes within the range [0, n_classes - 1].
    """

    if not isinstance(classes, List):
        raise ValueError(f"Classes should be a list, got {type(classes)}.")

    if isinstance(scores, List):
        scores = np.array(scores)

    if len(scores.shape) != 2:
        raise ValueError(f"Scores should be a 2D array, got {scores.shape}.")

    if scores.shape[1] != len(classes):
        raise ValueError(
            f"Number of labels and scores mismatch. Provided {len(classes)} class names and {scores.shape[1]} scores."
        )

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Scores should be in the range [0, 1].")

    if np.any(np.isclose(scores.sum(axis=1), 1.0, atol=1e-2)):
        raise ValueError("Each row of scores should sum to 1.")

    if ignored_indexes is not None:
        if not isinstance(ignored_indexes, List):
            raise ValueError(
                f"Ignored indexes should be a list, got {type(ignored_indexes)}."
            )
        if np.any(np.array(ignored_indexes) < 0) or np.any(
            np.array(ignored_indexes) >= len(classes)
        ):
            raise ValueError(
                "Ignored indexes should be integers in the range [0, num_classes -1]."
            )

    selection = np.ones(len(scores), dtype=bool)

    indexes = np.argmax(scores, axis=1)

    if remove_duplicates:
        selection[1:] = indexes[1:] != indexes[:-1]

    if ignored_indexes is not None:
        selection &= indexes != ignored_indexes

    class_list = [classes[i] for i in indexes[selection]]
    score_list = scores[selection].tolist()

    classification_msg = Classifications()

    classification_msg.classes = class_list
    classification_msg.scores = score_list

    return classification_msg
