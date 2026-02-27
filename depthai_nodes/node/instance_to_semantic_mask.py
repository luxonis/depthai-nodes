import depthai as dai
import numpy as np

from depthai_nodes.message import ImgDetectionsExtended
from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class InstanceToSemanticMask(BaseHostNode):
    """Converts an ImgDetectionsExtended or dai.ImgDetections instance mask into a
    semantic mask by mapping unique instance IDs to detection class labels.

    Attributes
    ----------
    detections: ImgDetectionsExtended or dai.ImgDetections
        Input detections with instance segmentation masks.
    out: ImgDetectionsExtended or dai.ImgDetections
        Output detections with semantic segmentation masks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgDetections, True)])

    def build(self, detections: dai.Node.Output) -> "InstanceToSemanticMask":
        self.link_args(detections)
        return self

    def process(self, msg: dai.Buffer) -> None:
        if not isinstance(msg, ImgDetectionsExtended) and not isinstance(
            msg, dai.ImgDetections
        ):
            raise TypeError(
                f"Expected ImgDetectionsExtended or dai.ImgDetections input type, got {type(msg)}."
            )

        msg_copy = copy_message(msg)
        if isinstance(msg_copy, ImgDetectionsExtended):
            masks = msg_copy.masks
        elif isinstance(msg_copy, dai.ImgDetections):
            masks = msg_copy.getCvSegmentationMask()

        if masks is None or masks.size == 0:
            self.out.send(msg_copy)
            return

        dets = msg_copy.detections
        if dets:
            # Lookup table (instance_id -> class_label) for vectorized mask remapping
            lut = np.array([int(det.label) for det in dets], dtype=np.int16)

            if isinstance(msg_copy, ImgDetectionsExtended):
                semantic_mask = np.full(masks.shape, -1, dtype=np.int16)
                mask_valid = (masks >= 0) & (masks < lut.size)

            elif isinstance(msg_copy, dai.ImgDetections):
                semantic_mask = np.full(masks.shape, 255, dtype=np.uint8)
                mask_valid = (masks < 255) & (masks < lut.size)

            if np.any(mask_valid):
                semantic_mask[mask_valid] = lut[masks[mask_valid]]

            if isinstance(msg_copy, ImgDetectionsExtended):
                msg_copy.masks = semantic_mask
            else:
                msg_copy.setCvSegmentationMask(semantic_mask)
        self.out.send(msg_copy)
