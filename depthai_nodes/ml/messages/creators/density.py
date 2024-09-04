import depthai as dai
import numpy as np

from ...messages import Map2D

def create_density_message(
    density_map: np.ndarray
) -> dai.ImgFrame:
    """Create a DepthAI message for a density map.

    @param density_map: A NumPy array representing the density map with shape HW or NHW/HWN.
        Here N stands for batch dimension.
    @type density_map: np.array
    @return: An Map2D object containing the density information.
    @rtype: Map2D
    @raise ValueError: If the density map is not a NumPy array.
    @raise ValueError: If the density map is not 2D or 3D.
    @raise ValueError: If the 3D density map shape is not NHW or HWN.
    """

    if not isinstance(density_map, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(density_map)}.")

    if not (len(density_map.shape) == 2 or len(density_map.shape) == 3) :
        raise ValueError(f"Expected 2D or 3D input, got {len(density_map.shape)}D input.")
    
    if len(density_map.shape) == 3:
        if density_map.shape[0] == 1:
            density_map = density_map[0, :, :]  # NHW to HW
        elif density_map.shape[2] == 1:
            density_map = density_map[:, :, 0]  # HWN to HW
        else:
            raise ValueError(
                f"Unexpected density map shape. Expected NHW or HWN, got {density_map.shape}."
            )

    map = Map2D()
    map.map = density_map

    return map