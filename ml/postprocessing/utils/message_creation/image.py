import depthai as dai
import numpy as np
import cv2


def create_image_msg(
    image: np.array,
    is_grayscale: bool = False,
    is_hwc: bool = True,
    is_bgr: bool = True,
) -> dai.ImgFrame:
    """
    Create a depthai message for an image array.

    @type image: np.array
    @ivar image: Image array.

    @type is_grayscale: bool
    @ivar is_grayscale: If True, the image is in grayscale format.

    @type is_hwc: bool
    @ivar is_hwc: If True, the image is in HWC format. If False, the image is in CHW format.

    @type is_bgr: bool
    @ivar is_bgr: If True, the image is in BGR format. If False, the image is in RGB format.
    """

    if is_grayscale:
        img_frame_type = dai.ImgFrame.Type.GRAY8  # HW image
        height, width = image.shape
    else:
        img_frame_type = dai.ImgFrame.Type.BGR888i  # HWC BGR image
        if not is_hwc:
            image = np.transpose(image, (1, 2, 0))
        if not is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(image)
    imgFrame.setWidth(width)
    imgFrame.setHeight(height)
    imgFrame.setType(img_frame_type)

    return imgFrame
