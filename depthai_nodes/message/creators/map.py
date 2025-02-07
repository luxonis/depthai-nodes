import numpy as np

from depthai_nodes import Map2D


def create_map_message(map: np.ndarray, min_max_scaling: bool = False) -> Map2D:
    """Create a DepthAI message for a map of floats.

    @param map: A NumPy array representing the map with shape HW or NHW/HWN. Here N
        stands for batch dimension.
    @type map: np.array
    @param min_max_scaling: If True, the map is scaled to the range [0, 1]. Defaults to
        False.
    @type min_max_scaling: bool
    @return: An Map2D object containing the density information.
    @rtype: Map2D
    @raise ValueError: If the density map is not a NumPy array.
    @raise ValueError: If the density map is not 2D or 3D.
    @raise ValueError: If the 3D density map shape is not NHW or HWN.
    """

    if not isinstance(map, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(map)}.")

    if not (len(map.shape) == 2 or len(map.shape) == 3):
        raise ValueError(f"Expected 2D or 3D input, got {len(map.shape)}D input.")

    if len(map.shape) == 3:
        if map.shape[0] == 1:
            map = map[0, :, :]  # NHW to HW
        elif map.shape[2] == 1:
            map = map[:, :, 0]  # HWN to HW
        else:
            raise ValueError(
                f"Unexpected map shape. Expected NHW or HWN, got {map.shape}."
            )

    if min_max_scaling:
        min_val = map.min()
        max_val = map.max()
        if map.min() != map.max():
            map = (map - min_val) / (max_val - min_val)

    if map.dtype != np.float32:
        map = map.astype(np.float32)

    map_2d = Map2D()
    map_2d.map = map

    return map_2d
