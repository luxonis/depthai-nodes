import numpy as np


def compute_regression_predictions(predictions: np.ndarray) -> list[float]:
    """Convert a regression tensor into a flat Python list."""
    squeezed = np.asarray(predictions).squeeze()
    return np.atleast_1d(squeezed).tolist()
