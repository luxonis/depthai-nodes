import depthai as dai
import numpy as np

import depthai_nodes.message.creators as creators

from .constants import ARRAYS


def create_img_frame(
    image: np.ndarray = ARRAYS["3d"],
    img_frame_type=dai.ImgFrame.Type.BGR888p,
):
    return creators.create_image_message(
        image=image.astype(np.uint8), img_frame_type=img_frame_type
    )


def create_map(map: np.ndarray = ARRAYS["2d"]):
    return creators.create_map_message(map=map.astype(np.float32))


def create_segmentation(mask: np.ndarray = ARRAYS["2d"]):
    return creators.create_segmentation_message(mask=mask.astype(np.int16))
