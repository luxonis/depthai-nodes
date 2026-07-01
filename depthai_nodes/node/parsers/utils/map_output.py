import numpy as np


def compute_map_output(map_tensor: np.ndarray) -> np.ndarray:
    """Return the model map output without the batch dimension."""
    map_output = np.asarray(map_tensor)
    if map_output.shape[0] == 1:
        map_output = map_output[0]
    return map_output
