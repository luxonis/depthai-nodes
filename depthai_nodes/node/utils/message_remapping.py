import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.utils.util_constants import GMessage


def remap_message(
    message: GMessage,
    from_transformation: dai.ImgTransformation | None,
    to_transformation: dai.ImgTransformation,
) -> GMessage:
    """Remap a transformable DepthAI message to a target image transformation.

    ``from_transformation`` remains part of the API for callers which resolve the
    source transformation explicitly. Native messages carry that transformation
    themselves. Native ``transformTo`` handles coordinates; pixel arrays are
    warped explicitly because native map and mask transformations do not do so.
    """

    if not isinstance(
        message,
        (
            dai.ImgDetections,
            dai.SegmentationMask,
            dai.beta.Keypoints,
            dai.beta.Clusters,
            dai.beta.Map2D,
            dai.beta.Lines,
            dai.beta.Predictions,
            dai.beta.Classifications,
        ),
    ):
        raise TypeError(
            f"Cannot remap message. Unsupported message type: {type(message)}"
        )

    if message.getTransformation() is None:
        if from_transformation is None:
            return message
        message.setTransformation(from_transformation)
    source = message.getTransformation()
    remapped = message.transformTo(to_transformation)
    # Native transformTo copies may share pixel storage with the source. Detach
    # before writing a differently sized array to preserve the original buffer.
    if isinstance(message, dai.beta.Map2D):
        remapped = copy_message(remapped)
        remapped.setMap(_remap_pixels(message.getMap(), source, to_transformation))
    elif isinstance(message, dai.SegmentationMask):
        remapped = copy_message(remapped)
        remapped.setCvMask(
            _remap_pixels(message.getCvMask(), source, to_transformation)
        )
    elif isinstance(message, dai.ImgDetections):
        mask = message.getCvSegmentationMask()
        if mask is not None and mask.size > 0:
            remapped = copy_message(remapped)
            remapped.setCvSegmentationMask(
                _remap_pixels(mask, source, to_transformation)
            )
    remapped.setTransformation(to_transformation)
    return remapped


def _remap_pixels(
    pixels: np.ndarray,
    source: dai.ImgTransformation,
    target: dai.ImgTransformation,
) -> np.ndarray:
    matrix = np.array(target.getMatrix()) @ np.array(source.getMatrixInv())
    # Preserve the previous background values and discrete mask labels.
    border_value = 255 if pixels.dtype == np.uint8 else -1
    return cv2.warpPerspective(
        pixels,
        matrix,
        target.getSize(),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
