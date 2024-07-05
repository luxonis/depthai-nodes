import cv2
import depthai as dai
import numpy as np


def create_image_message(
    image: np.array,
    is_bgr: bool = True,
) -> dai.ImgFrame:
    """Create a depthai message for an image array.

    @type image: np.array
    @ivar image: Image array in HWC or CHW format.
    @type is_bgr: bool
    @ivar is_bgr: If True, the image is in BGR format. If False, the image is in RGB
        format.
    """

    if image.shape[0] in [1, 3]:
        hwc = False
    elif image.shape[2] in [1, 3]:
        hwc = True
    else:
        raise ValueError(
            "Unexpected image shape. Expected CHW or HWC, got", image.shape
        )

    if not hwc:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[2] == 1:  # grayscale
        image = image[:, :, 0]
        img_frame_type = dai.ImgFrame.Type.GRAY8  # HW image
        height, width = image.shape
    else:
        if not is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_frame_type = dai.ImgFrame.Type.BGR888i  # HWC BGR image
        height, width, _ = image.shape

    imgFrame = dai.ImgFrame()
    imgFrame.setFrame(image)
    imgFrame.setWidth(width)
    imgFrame.setHeight(height)
    imgFrame.setType(img_frame_type)

    return imgFrame
