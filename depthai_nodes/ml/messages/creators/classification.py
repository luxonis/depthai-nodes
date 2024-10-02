from typing import List, Union

import numpy as np

from ...messages import Classifications, CompositeMessage


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

    if any([value < 0 or value > 1 for value in scores]):
        raise ValueError(
            f"Scores list must contain probabilities between 0 and 1, instead got {scores}."
        )

    if not np.isclose(np.sum(scores), 1.0, atol=1e-1):
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


def create_classification_sequence_message(
    classes: List[str],
    scores: Union[np.ndarray, List],
    ignored_indexes: List[int] = None,
    remove_duplicates: bool = False,
    concatenate_text: bool = False,
) -> Classifications:
    """Creates a message for a multi-class sequence. The 'scores' array is a sequence of
    probabilities for each class at each position in the sequence. The message contains
    the class names and their respective scores, ordered according to the sequence.

    @param classes: A list of class names, with length 'n_classes'.
    @type classes: List
    @param scores: A numpy array of shape (sequence_length, n_classes) containing the (row-wise) probability distributions over the classes.
    @type scores: np.ndarray
    @param ignored_indexes: A list of indexes to ignore during classification generation (e.g., background class, padding class)
    @type ignored_indexes: List[int]
    @param remove_duplicates: If True, removes consecutive duplicates from the sequence.
    @type remove_duplicates: bool
    @param concatenate_text: If True, concatenates consecutive words based on the space character.
    @type concatenate_text: bool
    @return: A Classification message with attributes `classes` and `scores`, where `classes` is a list of class names and `scores` is a list of corresponding scores.
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
            f"Number of classes and scores mismatch. Provided {len(classes)} class names and {scores.shape[1]} scores."
        )

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Scores should be in the range [0, 1].")

    if np.any(~np.isclose(scores.sum(axis=1), 1.0, atol=1e-2)):
        raise ValueError("Each row of scores should sum to 1.")

    if ignored_indexes is not None:
        if not isinstance(ignored_indexes, List):
            raise ValueError(
                f"Ignored indexes should be a list, got {type(ignored_indexes)}."
            )
        if not all(isinstance(index, int) for index in ignored_indexes):
            raise ValueError("Ignored indexes should be integers.")
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
        selection &= np.array([index not in ignored_indexes for index in indexes])

    class_list = [classes[i] for i in indexes[selection]]
    score_list = np.max(scores, axis=1)[selection]

    if (
        concatenate_text
        and len(class_list) > 1
        and all(len(word) <= 1 for word in class_list)
    ):
        concatenated_scores = []
        concatenated_words = "".join(class_list).split()
        cumsumlist = np.cumsum([len(word) for word in concatenated_words])

        start_index = 0
        for num_spaces, end_index in enumerate(cumsumlist):
            word_scores = score_list[start_index + num_spaces : end_index + num_spaces]
            concatenated_scores.append(np.mean(word_scores))
            start_index = end_index

        class_list = concatenated_words
        score_list = np.array(concatenated_scores)

    elif (
        concatenate_text
        and len(class_list) > 1
        and any(len(word) >= 2 for word in class_list)
    ):
        class_list = [" ".join(class_list)]
        score_list = np.mean(score_list)

    classification_msg = Classifications()

    classification_msg.classes = class_list
    classification_msg.scores = score_list.tolist()

    return classification_msg
