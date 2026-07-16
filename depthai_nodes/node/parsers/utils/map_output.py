import numpy as np


def compute_map_output(map_tensor: np.ndarray) -> np.ndarray:
    """Return the model map output without the batch dimension."""
    map_output = np.asarray(map_tensor)

    while map_output.ndim > 2 and map_output.shape[0] == 1:
        map_output = map_output[0]

    if map_output.ndim == 2:
        return map_output

    if map_output.ndim == 3 and map_output.shape[-1] == 1:
        return map_output[:, :, 0]

    raise ValueError(
        f"Expected HW, NHW, or HWN with singleton N; got {map_output.shape}."
    )
