from typing import Literal
import depthai as dai
import numpy as np

UINT16_MAX_VALUE = 65535


def create_monocular_depth_msg(
    depth_map: np.array, depth_type: Literal["relative", "metric"]
) -> dai.ImgFrame:
    """
    Creates a depth message in the form of an ImgFrame using the provided depth map and depth type.

    Args:
        depth_map (np.array): A NumPy array representing the depth map with shape (height, width).
        depth_type (Literal['relative', 'metric']): A string indicating the type of depth map.
            It can either be 'relative' or 'metric'.

    Returns:
        dai.ImgFrame: An ImgFrame object containing the depth information.

    """

    if depth_type == "relative":
        data_type = dai.ImgFrame.Type.RAW16

        # normalize depth map to the range [0, 65535]
        min_val = depth_map.min()
        max_val = depth_map.max()
        if min_val == max_val:  # avoid division by zero
            depth_map = np.zeros_like(depth_map)
        else:
            depth_map = (depth_map - min_val) / (max_val - min_val) * UINT16_MAX_VALUE
        depth_map = depth_map.astype(np.uint16)

    elif depth_type == "metric":
        raise NotImplementedError(
            "The message for 'metric' depth type is not yet implemented."
        )
    else:
        raise ValueError(
            f"Invalid depth type: {depth_type}. Only 'relative' and 'metric' are supported."
        )

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(depth_map)
    imgFrame.setWidth(depth_map.shape[1])
    imgFrame.setHeight(depth_map.shape[0])
    imgFrame.setType(data_type)

    return imgFrame
