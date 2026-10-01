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
    """Remap a supported DepthAI message into a target image transformation.

    Native ``transformTo`` maps coordinate fields. Map and mask pixel arrays are warped
    separately with nearest-neighbor interpolation and detached from the source buffer
    before writing. Pixels outside the source image are filled with 255 for ``uint8``
    arrays and -1 for other dtypes.

    Args:
        message: An ``ImgDetections``, ``SegmentationMask``, or beta ``Keypoints``,
            ``Clusters``, ``Map2D``, ``Lines``, ``Predictions``, or ``Classifications``
            message.
        from_transformation: Fallback source transformation. If the message has no
            transformation, this value is attached to it before remapping. An existing
            message transformation takes precedence.
        to_transformation: Target coordinate space and output pixel dimensions.

    Returns:
        Remapped message carrying the target transformation. If neither the message nor
        ``from_transformation`` provides a source transformation, returns the original
        message unchanged.

    Raises:
        TypeError: If the message type is unsupported.
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
    """Warp a pixel array between image transformations.

    Args:
        pixels: Source map or mask, whose resolution may differ from ``source``.
        source: Transformation describing the source coordinate space.
        target: Transformation defining the output space and image size.

    Returns:
        Warped array using nearest-neighbor interpolation, with 255 as the border value
        for ``uint8`` arrays and -1 for other dtypes.
    """
    height, width = pixels.shape[:2]
    source_width, source_height = source.getSize()
    # Pixel arrays may have a different resolution than their transformation.
    pixel_to_source = np.diag([source_width / width, source_height / height, 1])
    matrix = (
        np.array(target.getMatrix()) @ np.array(source.getMatrixInv()) @ pixel_to_source
    )
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
