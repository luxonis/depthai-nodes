import cv2
import depthai as dai
import numpy as np

from .colors import get_adas_colors, get_ewasr_colors, get_selfie_colors
from .messages import parse_segmentation_message


def visualize_segmentation(
    frame: dai.ImgFrame, message: dai.ImgFrame, extraParams: dict
):
    mask = parse_segmentation_message(message)
    frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    n_classes = extraParams.get("n_classes", None)

    if n_classes is None:
        raise ValueError("Number of classes not provided in NN archive metadata.")

    if n_classes == 2:
        COLORS = get_selfie_colors()
    elif n_classes == 3:
        COLORS = get_ewasr_colors()
    else:
        COLORS = get_adas_colors()

    colormap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(COLORS):
        m = mask == class_id
        colormap[m] = color

    alpha = 0.5
    overlay = cv2.addWeighted(colormap, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Segmentation", overlay)

    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
