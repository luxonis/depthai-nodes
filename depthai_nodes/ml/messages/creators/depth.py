from typing import Literal

import depthai as dai
import numpy as np

UINT16_MAX_VALUE = 65535


def create_depth_message(
    depth_map: np.ndarray,
    depth_type: Literal["relative", "metric"],
    depth_limit: float = 0.0,
) -> dai.ImgFrame:
    """Create a DepthAI message for a depth map.

    @param depth_map: A NumPy array representing the depth map with shape HW or NHW/HWN.
        Here N stands for batch dimension.
    @type depth_map: np.array
    @param depth_type: A string indicating the type of depth map. It can either be
        'relative' or 'metric'.
    @type depth_type: Literal['relative', 'metric']
    @param depth_limit: The maximum depth value (in meters) to be used in the depth map.
        The default value is 0, which means no limit.
    @type depth_limit: float
    @return: An ImgFrame object containing the depth information.
    @rtype: dai.ImgFrame
    @raise ValueError: If the depth map is not a NumPy array.
    @raise ValueError: If the depth map is not 2D or 3D.
    @raise ValueError: If the depth map shape is not NHW or HWN.
    @raise ValueError: If the depth type is not 'relative' or 'metric'.
    @raise ValueError: If the depth limit is not 0 and the depth type is 'relative'.
    @raise ValueError: If the depth limit is 0 and the depth type is 'metric'.
    @raise ValueError: If the depth limit is negative.
    """

    if not isinstance(depth_map, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(depth_map)}.")

    if len(depth_map.shape) == 3:
        if depth_map.shape[0] == 1:
            depth_map = depth_map[0, :, :]  # NHW to HW
        elif depth_map.shape[2] == 1:
            depth_map = depth_map[:, :, 0]  # HWN to HW
        else:
            raise ValueError(
                f"Unexpected image shape. Expected NHW or HWN, got {depth_map.shape}."
            )

    if len(depth_map.shape) != 2:
        raise ValueError(f"Expected 2D or 3D input, got {len(depth_map.shape)}D input.")

    if not (depth_type == "relative" or depth_type == "metric"):
        raise ValueError(
            f"Invalid depth type: {depth_type}. Only 'relative' and 'metric' are supported."
        )

    if depth_type == "relative" and depth_limit != 0:
        raise ValueError(
            f"Invalid depth limit: {depth_limit}. For relative depth, depth limit must be equal to 0."
        )

    if depth_type == "metric" and depth_limit == 0:
        raise ValueError(
            f"Invalid depth limit: {depth_limit}. For metric depth, depth limit must be bigger than 0."
        )

    if depth_limit < 0:
        raise ValueError(
            f"Invalid depth limit: {depth_limit}. Depth limit must be bigger than 0."
        )

    data_type = dai.ImgFrame.Type.RAW16

    min_val = depth_map.min() if depth_type == "relative" else 0
    max_val = depth_map.max() if depth_type == "relative" else depth_limit

    # clip values bigger than max_val
    depth_map = np.clip(depth_map, a_min=None, a_max=max_val)

    # normalize depth map to UINT16 range [0, UINT16_MAX_VALUE]
    if min_val == max_val:  # avoid division by zero
        depth_map = np.zeros_like(depth_map)
    else:
        depth_map = (depth_map - min_val) / (max_val - min_val) * UINT16_MAX_VALUE
    depth_map = depth_map.astype(np.uint16)

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(depth_map)
    imgFrame.setWidth(depth_map.shape[1])
    imgFrame.setHeight(depth_map.shape[0])
    imgFrame.setType(data_type)

    return imgFrame
