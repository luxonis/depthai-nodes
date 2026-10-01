import depthai as dai
import numpy as np


def create_map_message(
    map_array: np.ndarray, min_max_scaling: bool = False
) -> dai.beta.Map2D:
    """Create a float32 native map from a two-dimensional array.

    Args:
        map_array: NumPy array of shape ``(H, W)``, ``(1, H, W)``, or ``(H, W, 1)``. A
            singleton leading or trailing axis is removed.
        min_max_scaling: Scale nonconstant maps to [0, 1] when true. Constant maps keep
            their original values.

    Returns:
        A native ``Map2D`` message containing the HW float32 map.

    Raises:
        ValueError: If the input is not a NumPy array or has an unsupported shape.
    """

    if not isinstance(map_array, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(map_array)}.")

    if not (len(map_array.shape) == 2 or len(map_array.shape) == 3):
        raise ValueError(f"Expected 2D or 3D input, got {len(map_array.shape)}D input.")

    if len(map_array.shape) == 3:
        if map_array.shape[0] == 1:
            map_array = map_array[0, :, :]  # NHW to HW
        elif map_array.shape[2] == 1:
            map_array = map_array[:, :, 0]  # HWN to HW
        else:
            raise ValueError(
                f"Unexpected map shape. Expected NHW or HWN, got {map_array.shape}."
            )

    if min_max_scaling:
        min_val = map_array.min()
        max_val = map_array.max()
        if min_val != max_val:
            map_array = (map_array - min_val) / (max_val - min_val)

    if map_array.dtype != np.float32:
        map_array = map_array.astype(np.float32)

    map_2d = dai.beta.Map2D()
    map_2d.setMap(map_array)

    return map_2d
