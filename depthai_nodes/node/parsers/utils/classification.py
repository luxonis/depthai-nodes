import numpy as np

from .activations import softmax


def compute_classification_scores(
    scores: np.ndarray,
    *,
    is_softmax: bool = True,
) -> np.ndarray:
    """Flatten class scores and optionally apply softmax.

    Args:
        scores: Model score tensor.
        is_softmax: Whether scores already contain probabilities. If false, apply
            softmax.

    Returns:
        One-dimensional array with one score per class.
    """
    computed_scores = scores.flatten()
    if not is_softmax:
        computed_scores = softmax(computed_scores)
    return computed_scores
