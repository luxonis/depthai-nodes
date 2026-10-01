import depthai as dai
import numpy as np

from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class InstanceToSemanticMask(BaseHostNode):
    """Replace instance-mask IDs with detection class labels.

    The input and output are ``dai.ImgDetections`` messages. A copied message is
    emitted; its mask is replaced when both a mask and detections are present.
    Class labels must be representable in uint8, with 255 reserved for background.
    Missing or empty masks pass through on the copied message unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgDetections, True)])

    def build(self, detections: dai.Node.Output) -> "InstanceToSemanticMask":
        """Connect a detection stream with instance masks.

        Args:
            detections: Output producing ``dai.ImgDetections``.

        Returns:
            This node.
        """
        self.link_args(detections)
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Copy detections and convert valid mask indexes to class labels.

        Args:
            msg: Detections with a mask of non-negative instance indexes. Values of 255
                or indexes beyond the detection list become background.

        Raises:
            TypeError: If the input is not ``dai.ImgDetections``.
        """
        if not isinstance(msg, dai.ImgDetections):
            raise TypeError(f"Expected dai.ImgDetections input type, got {type(msg)}.")

        msg_copy = copy_message(msg)
        masks = msg_copy.getCvSegmentationMask()

        if masks is None or masks.size == 0:
            self.out.send(msg_copy)
            return

        dets = msg_copy.detections
        if dets:
            # Lookup table (instance_id -> class_label) for vectorized mask remapping
            lut = np.array([int(det.label) for det in dets], dtype=np.int16)

            semantic_mask = np.full(masks.shape, 255, dtype=np.uint8)
            mask_valid = (masks < 255) & (masks < lut.size)

            if np.any(mask_valid):
                semantic_mask[mask_valid] = lut[masks[mask_valid]]

            msg_copy.setCvSegmentationMask(semantic_mask)
        self.out.send(msg_copy)
