from typing import List

import depthai as dai

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node.utils import copy_message


class ImgDetectionsFilter(dai.node.HostNode):
    """Filters out detections that do not meet the specified criteria.

    Attributes
    ----------
    labels_to_keep : List[int]
        Labels to keep. Only detections with labels in this list will be kept.
    labels_to_reject: List[int]
        Labels to reject. Only detections with labels not in this list will be kept.
    confidence_threshold : float
        Minimum confidence threshold. Detections with confidence below this threshold will be filtered out.
    max_detections : int
        Maximum number of detections to keep. If not defined, all detections will be kept.
    """

    def __init__(self):
        super().__init__()
        self._labels_to_keep = None
        self._labels_to_reject = None
        self._confidence_threshold = None
        self._max_detections = None

    def setLabels(self, labels: List[int], keep: bool) -> None:
        """Sets the labels to keep or reject.

        @param labels: The labels to keep or reject.
        @type labels: List[int]
        @param keep: Whether to keep or reject the labels.
        @type keep: bool
        """
        if not isinstance(labels, list):
            raise ValueError("Labels must be a list.")
        if not isinstance(keep, bool):
            raise ValueError("keep must be a boolean.")

        if keep:
            self._labels_to_keep = labels
            self._labels_to_reject = None
        else:
            self._labels_to_keep = None
            self._labels_to_reject = labels

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        """Sets the confidence threshold.

        @param confidence_threshold: The confidence threshold.
        @type confidence_threshold: float
        """
        if not isinstance(confidence_threshold, float):
            raise ValueError("confidence_threshold must be a float.")
        self._confidence_threshold = confidence_threshold

    def setMaxDetections(self, max_detections: int) -> None:
        """Sets the maximum number of detections.

        @param max_detections: The maximum number of detections.
        @type max_detections: int
        """
        if not isinstance(max_detections, int):
            raise ValueError("max_detections must be an integer.")
        self._max_detections = max_detections

    def build(
        self,
        msg: dai.Node.Output,
        labels_to_keep: List[int] = None,
        labels_to_reject: List[int] = None,
        confidence_threshold: float = None,
        max_detections: int = None,
    ) -> "ImgDetectionsFilter":
        self.link_args(msg)

        if labels_to_keep is not None:
            if labels_to_reject is not None:
                raise ValueError(
                    "labels_to_keep and labels_to_reject cannot be set at the same time"
                )
            else:
                self.setLabels(labels_to_keep, keep=True)
        else:
            if labels_to_reject is not None:
                self.setLabels(labels_to_reject, keep=False)

        if confidence_threshold is not None:
            self.setConfidenceThreshold(confidence_threshold)

        if max_detections is not None:
            self.setMaxDetections(max_detections)

        return self

    def process(self, msg: dai.Buffer) -> None:
        assert isinstance(
            msg,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
            ),  # TODO: also allow dai.SpatialImgDetections?
        )

        msg_new = copy_message(
            msg
        )  # we don't want to modify the original message as it might be used by other nodes

        filtered_detections = []
        for detection in msg.detections:
            if self._labels_to_keep is not None:
                if detection.label not in self._labels_to_keep:
                    continue

            if self._labels_to_reject is not None:
                if detection.label in self._labels_to_reject:
                    continue

            if self._confidence_threshold is not None:
                if detection.confidence < self._confidence_threshold:
                    continue

            filtered_detections.append(detection)

        msg_new.detections = filtered_detections[
            : self._max_detections
        ]  # TODO: sort detections by confidence before subsetting?

        self.out.send(msg_new)
