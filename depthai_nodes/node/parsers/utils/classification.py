import numpy as np

from .activations import softmax


def compute_classification_scores(
    scores: np.ndarray,
    *,
    is_softmax: bool = True,
) -> np.ndarray:
    """Return classification scores, applying softmax when needed."""
    computed_scores = np.asarray(scores).flatten()
    if not is_softmax:
        computed_scores = softmax(computed_scores)
    return computed_scores
