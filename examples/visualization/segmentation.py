import cv2
import depthai as dai
import numpy as np

from depthai_nodes.ml.messages import SegmentationMasks

from .colors import get_adas_colors, get_ewasr_colors, get_selfie_colors
from .messages import parse_fast_sam_message, parse_segmentation_message


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


def _fastsam_show_masks(
    annotation,
    image,
):
    n, h, w = annotation.shape  # batch, height, width
    areas = np.sum(annotation, axis=(1, 2))
    annotation = annotation[np.argsort(areas)]

    index = (annotation != 0).argmax(axis=0)
    color = np.random.random((n, 1, 1, 3)) * 255
    transparency = np.ones((n, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    show = np.zeros((h, w, 4), dtype=np.float32)
    h_indices, w_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]

    overlay = image.astype(np.float32)
    alpha = show[:, :, 3:4]
    overlay[:, :, :3] = overlay[:, :, :3] * (1 - alpha) + show[:, :, :3] * alpha

    return overlay.astype(np.uint8)


def visualize_fastsam(
    frame: dai.ImgFrame, message: SegmentationMasks, extraParams: dict
):
    masks = parse_fast_sam_message(message)

    if masks is not None:
        for i, mask in enumerate(masks):
            masks[i] = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
            )

        overlay = _fastsam_show_masks(masks, frame)
    else:
        overlay = frame

    cv2.imshow("FastSAM segmentation", overlay)

    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return True

    return False
