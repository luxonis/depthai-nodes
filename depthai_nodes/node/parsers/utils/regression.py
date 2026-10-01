import numpy as np


def compute_regression_predictions(predictions: np.ndarray) -> list[float]:
    """Remove singleton axes from a regression tensor.

    Args:
        predictions: Model prediction tensor.

    Returns:
        A Python list obtained after squeezing singleton dimensions. Scalar predictions
        become a one-item list; remaining non-singleton dimensions produce nested lists.
    """
    squeezed = np.asarray(predictions).squeeze()
    return np.atleast_1d(squeezed).tolist()
