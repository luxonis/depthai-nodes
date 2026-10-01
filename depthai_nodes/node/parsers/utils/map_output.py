import numpy as np


def compute_map_output(map_tensor: np.ndarray) -> np.ndarray:
    """Remove singleton batch or channel axes from a numeric map.

    Args:
        map_tensor: HW map, a map with leading singleton axes, or an HW1 map.

    Returns:
        A two-dimensional array. Values and dtype are preserved; the result may share
        input storage.

    Raises:
        ValueError: If the input cannot be reduced to an HW map through supported
            singleton axes.
    """
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
