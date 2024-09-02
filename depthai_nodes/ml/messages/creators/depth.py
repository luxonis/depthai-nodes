from typing import Literal

import depthai as dai
import numpy as np

UINT16_MAX_VALUE = 65535


def create_depth_message(
    depth_map: np.ndarray, depth_type: Literal["relative", "metric"]
) -> dai.ImgFrame:
    """Create a DepthAI message for a depth map.

    @param depth_map: A NumPy array representing the depth map with shape (HW).
    @type depth_map: np.array
    @param depth_type: A string indicating the type of depth map. It can either be
        'relative' or 'metric'.
    @type depth_type: Literal['relative', 'metric']
    @return: An ImgFrame object containing the depth information.
    @rtype: dai.ImgFrame
    @raise ValueError: If the depth map is not a NumPy array.
    @raise ValueError: If the depth map is not 2D.
    @raise ValueError: If the depth type is not 'relative' or 'metric'.
    """

    if not isinstance(depth_map, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(depth_map)}.")

    if len(depth_map.shape) == 3:
        if depth_map.shape[0] == 1:
            depth_map = depth_map[0, :, :]  # CHW to HW
        elif depth_map.shape[2] == 1:
            depth_map = depth_map[:, :, 0]  # HWC to HW
        else:
            raise ValueError(
                f"Unexpected image shape. Expected CHW or HWC, got {depth_map.shape}."
            )

    if len(depth_map.shape) != 2:
        raise ValueError(f"Expected 2D or 3D input, got {len(depth_map.shape)}D input.")

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
