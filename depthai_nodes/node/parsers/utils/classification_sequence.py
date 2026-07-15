import numpy as np

from .activations import softmax


def compute_classification_sequence_scores(
    scores: np.ndarray,
    *,
    is_softmax: bool = True,
) -> np.ndarray:
    """Return per-step classification scores, applying softmax when needed."""
    computed_scores = np.asarray(scores, dtype=np.float32)

    if computed_scores.ndim not in (2, 3):
        raise ValueError(
            f"Scores should be a 3D or 2D array, got shape {computed_scores.shape}."
        )

    if computed_scores.ndim == 3:
        if computed_scores.shape[0] == 1:
            computed_scores = computed_scores[0]
        elif computed_scores.shape[2] == 1:
            computed_scores = computed_scores[:, :, 0]
        else:
            raise ValueError(
                "Scores should be a 3D array of shape "
                "(1, sequence_length, n_classes) or "
                "(sequence_length, n_classes, 1)."
            )

    if not is_softmax:
        computed_scores = softmax(computed_scores, axis=1, keep_dims=True)

    return computed_scores
