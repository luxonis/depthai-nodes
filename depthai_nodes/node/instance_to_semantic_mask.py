import numpy as np
import depthai as dai

from depthai_nodes.message import ImgDetectionsExtended
from depthai_nodes.message.utils import copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class InstanceToSemanticMask(BaseHostNode):
    """
    Converts an ImgDetectionsExtended instance mask into a semantic mask by
    mapping unique instance IDs to detection class labels.

    Attributes
    ----------
    detections: ImgDetectionsExtended
        Input detections with instance segmentation masks.
    out: ImgDetectionsExtended
        Output detections with semantic segmentation masks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgDetections, True)])

    def build(self, detections: dai.Node.Output) -> "InstanceToSemanticMask":
        self.link_args(detections)
        return self

    def process(self, msg: dai.Buffer) -> None:
        if not isinstance(msg, ImgDetectionsExtended):
            raise TypeError(f"Expected ImgDetectionsExtended input type, got {type(msg)}.")

        msg_copy = copy_message(msg)
        masks = msg_copy.masks
        if masks is None or masks.size == 0:
            self.out.send(msg_copy)
            return

        dets = msg_copy.detections
        if dets:
            # Lookup table (instance_id -> class_label) for vectorized mask remapping
            lut = np.array([int(det.label) for det in dets], dtype=np.int16)

            semantic_mask = np.full(masks.shape, -1, dtype=np.int16)
            mask_valid = (masks >= 0) & (masks < lut.size)
            if np.any(mask_valid):
                semantic_mask[mask_valid] = lut[masks[mask_valid]]

            msg_copy.masks = semantic_mask

        self.out.send(msg_copy)
