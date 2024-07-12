import depthai as dai
import numpy as np


def create_thermal_message(thermal_image: np.array) -> dai.ImgFrame:
    """Creates a thermal image message in the form of an ImgFrame using the provided
    thermal image array.

    Args:
        thermal_image (np.array): A NumPy array representing the thermal image with shape (CHW or HWC).

    Returns:
        dai.ImgFrame: An ImgFrame object containing the thermal information.
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
            "Unexpected image shape. Expected CHW or HWC, got", thermal_image.shape
        )

    thermal_image = thermal_image.astype(np.uint16)

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(thermal_image)
    imgFrame.setWidth(thermal_image.shape[1])
    imgFrame.setHeight(thermal_image.shape[0])
    imgFrame.setType(dai.ImgFrame.Type.RAW16)

    return imgFrame
