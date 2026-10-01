import cv2
import depthai as dai
import numpy as np


def create_image_message(
    image: np.ndarray,
    is_bgr: bool = True,
    img_frame_type: dai.ImgFrame.Type = dai.ImgFrame.Type.BGR888i,
) -> dai.ImgFrame:
    """Create an ImgFrame from an integer image array.

    Args:
        image: Non-empty CHW or HWC image with one or three channels. When both layouts
            are plausible, a leading size of 1 or 3 selects CHW. Floating-point image
            values are rejected.
        is_bgr: Whether three-channel input is BGR. False converts RGB input to BGR.
        img_frame_type: Requested DepthAI frame type. Grayscale input switches to GRAY8
            unless the requested type begins with RAW or GRAY.

    Returns:
        Image message with payload, dimensions, and frame type set.

    Raises:
        ValueError: If neither channel layout is recognized or the image contains
            floating-point values.
    """

    if image.shape[0] in [1, 3]:
        hwc = False
    elif image.shape[2] in [1, 3]:
        hwc = True
    else:
        raise ValueError(
            f"Unexpected image shape. Expected CHW or HWC, got {image.shape}"
        )

    if not hwc:
        image = np.transpose(image, (1, 2, 0))

    if isinstance(image[0, 0, 0], (float, np.floating)):
        raise ValueError(f"Expected int type, got {type(image[0, 0, 0])}.")

    if image.shape[2] == 1:  # grayscale
        image = image[:, :, 0]  # HW image
        if not (img_frame_type.name.startswith(("RAW", "GRAY"))):
            img_frame_type = dai.ImgFrame.Type.GRAY8
        height, width = image.shape
    else:
        if not is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

    imgFrame = dai.ImgFrame()
    imgFrame.setCvFrame(image, img_frame_type)
    imgFrame.setWidth(width)
    imgFrame.setHeight(height)
    imgFrame.setType(img_frame_type)

    return imgFrame
