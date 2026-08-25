import depthai as dai

from depthai_nodes.node.utils.util_constants import GMessage


def remap_message(
    message: GMessage,
    from_transformation: dai.ImgTransformation | None,
    to_transformation: dai.ImgTransformation,
) -> GMessage:
    """Remap a transformable DepthAI message to a target image transformation.

    ``from_transformation`` remains part of the API for callers which resolve the
    source transformation explicitly. Native messages carry that transformation
    themselves, so remapping is delegated to their ``transformTo`` implementation.
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
    return message.transformTo(to_transformation)
