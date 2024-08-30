import depthai as dai
import numpy as np


def create_thermal_message(thermal_image: np.ndarray) -> dai.ImgFrame:
    """Create a DepthAI message for thermal image.

    @param thermal_image: A NumPy array representing the thermal image with shape (CHW
        or HWC).
    @type thermal_image: np.array
    @return: An ImgFrame object containing the thermal information.
    @rtype: dai.ImgFrame
    @raise ValueError: If the input is not a NumPy array.
    @raise ValueError: If the input is not 3D.
    @raise ValueError: If the input shape is not CHW or HWC.
    """

    if not isinstance(thermal_image, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(thermal_image)}.")
    if len(thermal_image.shape) != 3:
        raise ValueError(f"Expected 3D input, got {len(thermal_image.shape)}D input.")

    if thermal_image.shape[0] == 1:
        thermal_image = thermal_image[0, :, :]  # CHW to HW
    elif thermal_image.shape[2] == 1:
        thermal_image = thermal_image[:, :, 0]  # HWC to HW
    else:
        raise ValueError(
            f"Unexpected image shape. Expected CHW or HWC, got {thermal_image.shape}."
        )

    if isinstance(thermal_image.flat[0], float):
        raise ValueError("Expected integer values, got float.")

    if np.any(thermal_image < 0):
        raise ValueError("All values of thermal_image have to be non-negative.")

    thermal_image = thermal_image.astype(np.uint16)

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(thermal_image)
    imgFrame.setWidth(thermal_image.shape[1])
    imgFrame.setHeight(thermal_image.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW16)

    return imgFrame
