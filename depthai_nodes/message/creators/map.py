import depthai as dai
import numpy as np


def create_map_message(
    map_array: np.ndarray, min_max_scaling: bool = False
) -> dai.beta.Map2D:
    """Create a DepthAI message for a map of floats.

    @param map_array: A NumPy array representing the map with shape HW or NHW/HWN. Here
        N stands for batch dimension.
    @type map_array: np.array
    @param min_max_scaling: If True, the map is scaled to the range [0, 1]. Defaults to
        False.
    @type min_max_scaling: bool
    @return: A native Map2D object containing the density information.
    @rtype: dai.beta.Map2D
    @raise ValueError: If the density map is not a NumPy array.
    @raise ValueError: If the density map is not 2D or 3D.
    @raise ValueError: If the 3D density map shape is not NHW or HWN.
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
