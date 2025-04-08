import cv2
import depthai as dai
import numpy as np


def create_image_message(
    image: np.ndarray,
    is_bgr: bool = True,
    img_frame_type: dai.ImgFrame.Type = dai.ImgFrame.Type.BGR888i,
) -> dai.ImgFrame:
    """Create a DepthAI message for an image array.

    @param image: Image array in HWC or CHW format.
    @type image: np.array
    @param is_bgr: If True, the image is in BGR format. If False, the image is in RGB
        format. Defaults to True.
    @type is_bgr: bool
    @param img_frame_type: Output ImgFrame type. Defaults to BGR888i.
    @type img_frame_type: dai.ImgFrame.Type
    @return: dai.ImgFrame object containing the image information.
    @rtype: dai.ImgFrame
    @raise ValueError: If the image shape is not CHW or HWC.
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
