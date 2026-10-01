import depthai as dai
import numpy as np


def create_classification_message(
    classes: list[str], scores: np.ndarray | list
) -> dai.beta.Classifications:
    """Create a classification message sorted by descending score.

    Args:
        classes: Non-empty list of class names.
        scores: Floating-point probabilities corresponding to ``classes``. Values must
            be between 0 and 1 and sum to 1 within an absolute tolerance of 0.1. A list
            or array that can be flattened to one probability per class is accepted.

    Returns:
        Native classification message containing sorted class names and scores. Classes
        with equal scores retain their input order.

    Raises:
        ValueError: If classes or scores are empty, have unsupported types, differ in
            length, or scores are not valid floating-point probabilities.
    """
    if isinstance(classes, type(None)):
        raise ValueError("Classes should not be None.")

    if not isinstance(classes, list):
        raise ValueError(f"Classes should be a list, got {type(classes)}.")

    if len(classes) == 0:
        raise ValueError("Classes should not be empty.")

    if scores is None:
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

    scores = scores.astype(np.float32)

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

    classification_msg = dai.beta.Classifications()
    sorted_args = np.argsort(-scores, kind="stable")
    scores = scores[sorted_args]

    classification_msg.classes = [classes[i] for i in sorted_args]
    classification_msg.scores = scores

    return classification_msg


def create_classification_sequence_message(
    classes: list[str],
    scores: np.ndarray | list,
    ignored_indexes: list[int] | None = None,
    remove_duplicates: bool = False,
    concatenate_classes: bool = False,
) -> dai.beta.Classifications:
    """Create a classification sequence from per-position class probabilities.

    Args:
        classes: Class names, indexed by the columns of ``scores``.
        scores: Array or nested list of shape ``(sequence_length, n_classes)``. Each row
            must contain probabilities between 0 and 1 that sum to 1 within an absolute
            tolerance of 0.01.
        ignored_indexes: Class indexes to omit, such as a padding or background class.
        remove_duplicates: Remove adjacent repeated winning classes before filtering
            ignored indexes.
        concatenate_classes: If all selected class names have at most one character,
            join them into words separated by spaces and average the scores within each
            word.

    Returns:
        Native classification message with selected class names and scores in sequence
        order.

    Raises:
        ValueError: If classes are not a list, scores have incompatible dimensions or
            invalid probabilities, or ignored indexes are not a list of valid integer
            class indexes.
    """

    if not isinstance(classes, list):
        raise ValueError(f"Classes should be a list, got {type(classes)}.")

    if isinstance(scores, list):
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
        raise ValueError(
            f"Each row of scores should sum to 1, got {scores.sum(axis=1)}."
        )

    scores = scores.astype(np.float32)

    if ignored_indexes is not None:
        if not isinstance(ignored_indexes, list):
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

    class_list: list[str] = [classes[i] for i in indexes[selection]]
    score_list = np.max(scores, axis=1)[selection]

    if (
        concatenate_classes
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
        concatenate_classes
        and len(class_list) > 1
        and any(len(word) >= 2 for word in class_list)
    ):
        class_list = [" ".join(class_list)]
        mean_score = np.mean(score_list)
        score_list = np.array([mean_score])

    classification_msg = dai.beta.Classifications()

    classification_msg.classes = class_list
    classification_msg.scores = score_list

    return classification_msg
