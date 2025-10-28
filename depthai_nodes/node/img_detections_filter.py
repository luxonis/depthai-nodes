from typing import List

import depthai as dai

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.message.utils import compute_area, copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


class ImgDetectionsFilter(BaseHostNode):
    """Filters out detections based on the specified criteria and outputs them as a separate message.
    The order of operations:
        1. Filter by label/confidence/area;
        2. Sort (if applicable);
        3. Subset.

    Attributes
    ----------
    labels_to_keep : List[int]
        Labels to keep. Only detections with labels in this list will be kept.
    labels_to_reject: List[int]
        Labels to reject. Only detections with labels not in this list will be kept.
    confidence_threshold : float
        Minimum confidence threshold. Detections with confidence below this threshold will be filtered out.
    min_area : float
        Minimum normalized area (width * height) for a detection's bounding box.
    max_detections : int
        Maximum number of detections to keep. If not defined, all detections will be kept.
    sort_by_confidence: bool
        Whether to sort the detections by confidence before subsetting.
    """

    def __init__(self):
        super().__init__()
        self._labels_to_keep = None
        self._labels_to_reject = None
        self._confidence_threshold = None
        self._max_detections = None
        self._sort_by_confidence = False
        self._min_area = None
        self._logger.debug("ImgDetectionsFilter initialized")

    def setLabels(self, labels: List[int], keep: bool) -> None:
        if not isinstance(labels, list) and labels is not None:
            raise ValueError("Labels must be a list or None.")
        if not isinstance(keep, bool):
            raise ValueError("keep must be a boolean.")

        if keep:
            self._labels_to_keep = labels
            self._labels_to_reject = None
        else:
            self._labels_to_keep = None
            self._labels_to_reject = labels

        self._logger.debug(
            f"Labels set to {self._labels_to_keep if keep else self._labels_to_reject}"
        )

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        if (
            not isinstance(confidence_threshold, float)
            and confidence_threshold is not None
        ):
            raise ValueError("confidence_threshold must be a float.")
        self._confidence_threshold = confidence_threshold
        self._logger.debug(f"Confidence threshold set to {self._confidence_threshold}")

    def setMaxDetections(self, max_detections: int) -> None:
        if not isinstance(max_detections, int) and max_detections is not None:
            raise ValueError("max_detections must be an integer.")
        self._max_detections = max_detections
        self._logger.debug(f"Max detections set to {self._max_detections}")

    def setSortByConfidence(self, sort_by_confidence: bool) -> None:
        if not isinstance(sort_by_confidence, bool):
            raise ValueError("sort_by_confidence must be a boolean.")
        self._sort_by_confidence = sort_by_confidence
        self._logger.debug(f"Sort by confidence set to {self._sort_by_confidence}")

    def setMinArea(self, min_area: float) -> None:
        if min_area is not None:
            if not isinstance(min_area, (float)):
                raise ValueError("min_area must be a float.")
            if min_area < 0:
                raise ValueError("min_area must be non-negative.")
        self._min_area = float(min_area) if min_area is not None else None
        self._logger.debug(f"Minimum area set to {self._min_area}")

    def build(
        self,
        msg: dai.Node.Output,
        labels_to_keep: List[int] = None,
        labels_to_reject: List[int] = None,
        confidence_threshold: float = None,
        max_detections: int = None,
        sort_by_confidence: bool = False,
        min_area: float = None,
    ) -> "ImgDetectionsFilter":
        self.link_args(msg)

        if labels_to_keep is not None:
            if labels_to_reject is not None:
                raise ValueError(
                    "Cannot set labels_to_keep and labels_to_reject at the same time."
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

        if min_area is not None:
            self.setMinArea(min_area)

        self.setSortByConfidence(sort_by_confidence)

        self._logger.debug(
            f"ImgDetectionsFilter built with labels_to_keep={labels_to_keep}, "
            f"labels_to_reject={labels_to_reject}, confidence_threshold={confidence_threshold}, "
            f"max_detections={max_detections}, sort_by_confidence={sort_by_confidence}, "
            f"min_area={min_area}"
        )

        return self

    def process(self, msg: dai.Buffer) -> None:
        self._logger.debug("Processing new input")
        assert isinstance(
            msg,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
                dai.SpatialImgDetections,
            ),
        )

        msg_new = copy_message(msg)
        assert isinstance(
            msg_new,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
                dai.SpatialImgDetections,
            ),
        )

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

            if self._min_area is not None:
                area = compute_area(detection)
                if area < self._min_area:
                    continue

            filtered_detections.append(detection)

        if self._sort_by_confidence:
            filtered_detections = sorted(
                filtered_detections, key=lambda x: x.confidence, reverse=True
            )

        msg_new.detections = filtered_detections[: self._max_detections]

        self._logger.debug("Detections message created")
        self.out.send(msg_new)
        self._logger.debug("Message sent successfully")
