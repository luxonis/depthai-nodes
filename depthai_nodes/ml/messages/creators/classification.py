import numpy as np

from ...messages import Classifications


def create_classification_message(
    scores: np.ndarray, classes: np.ndarray = None
) -> Classifications:
    """Create a message for classification. The message contains the class names and
    their respective scores, sorted in descending order of scores.

    @param scores: A numpy array of shape (n_classes,) containing the probability score of each class.
    @type scores: np.ndarray
    @param classes: A numpy array of shape (n_classes, ), containing class names. If not provided, class names are set to [].
    @type classes: np.ndarray

    @return: A message with attributes `classes` and `scores`. `classes` is a list of classes, sorted in descending order of scores. `scores` is a list of the corresponding scores.
    @rtype: Classifications

    @raise ValueError: If the scores are empty.
    @raise ValueError: If the scores are not a 1D array.
    @raise ValueError: If the classes are not a 1D array.
    @raise ValueError: If the scores are not of type float.
    @raise ValueError: If the scores do not sum to 1.
    @raise ValueError: If the number of labels and scores mismatch.
    """

    if type(classes) == type(None):
        classes = np.array([])
    else:
        classes = np.array(classes)

    if len(scores) == 0:
        raise ValueError("Scores should not be empty.")

    if len(scores) != len(scores.flatten()):
        raise ValueError(f"Scores should be a 1D array, got {scores.shape}.")

    if len(classes) != len(classes.flatten()):
        raise ValueError(f"Classes should be a 1D array, got {classes.shape}.")

    scores = scores.flatten()
    classes = classes.flatten()

    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(f"Scores should be of type float, got {scores.dtype}.")

    if not np.isclose(np.sum(scores), 1.0, atol=1e-1):
        raise ValueError(f"Scores should sum to 1, got {np.sum(scores)}.")

    if len(scores) != len(classes) and len(classes) != 0:
        raise ValueError(
            f"Number of labels and scores mismatch. Provided {len(scores)} scores and {len(classes)} class names."
        )

    classification_msg = Classifications()

    sorted_args = np.argsort(scores)[::-1]
    scores = scores[sorted_args]

    if len(classes) != 0:
        classification_msg.classes = classes[sorted_args].tolist()

    classification_msg.scores = scores.tolist()

    return classification_msg
