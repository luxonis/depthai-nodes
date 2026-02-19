import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.utils import compute_area, copy_message
from depthai_nodes.node.base_host_node import BaseHostNode


@dataclass
class _FilterCfg:
    labels_to_keep: Optional[List[int]] = None
    labels_to_reject: Optional[List[int]] = None
    min_confidence: Optional[float] = None
    min_area: Optional[float] = None
    sort_desc: bool = True
    sort_disabled: bool = True
    first_k: Optional[int] = None


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
        self._cfg = _FilterCfg()
        self._logger.debug("ImgDetectionsFilter initialized")

    def setLabels(self, labels: List[int], keep: bool) -> None:
        warnings.warn(
            "setLabels() is deprecated; use keepLabels() or rejectLabels() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if keep:
            self.keepLabels(labels=labels)
        else:
            self.rejectLabels(labels=labels)

    def setConfidenceThreshold(self, confidence_threshold: float | None) -> None:
        warnings.warn(
            "setConfidenceThreshold() is deprecated; use minConfidence() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minConfidence(threshold=confidence_threshold)

    def setMaxDetections(self, max_detections: int) -> None:
        warnings.warn(
            "setMaxDetections() is deprecated; use takeFirstK() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.takeFirstK(k=max_detections)

    def setSortByConfidence(self, sort_by_confidence: bool) -> None:
        warnings.warn(
            "setSortByConfidence() is deprecated; use sortByConfidence(), enableSorting() and disableSorting() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if sort_by_confidence is True:
            self.enableSorting()
        else:
            self.disableSorting()

    def setMinArea(self, min_area: float) -> None:
        warnings.warn(
            "setMinArea() is deprecated; use minArea() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minArea(area=min_area)

    def keepLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        self._cfg.labels_to_keep = labels
        if self._cfg.labels_to_reject is not None:
            self._logger.warn(f"Removing labels to reject. Use either `keepLabels` or `rejectLabels` but not both.")
            self._cfg.labels_to_reject = None
        return self

    def rejectLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        self._cfg.labels_to_reject = labels
        if self._cfg.labels_to_keep is not None:
            self._logger.warn(f"Removing labels to keep. Use either `keepLabels` or `rejectLabels` but not both.")
            self._cfg.labels_to_keep = None
        return self

    def minConfidence(self, threshold: float) -> "ImgDetectionsFilter":
        self._cfg.min_confidence = threshold
        return self

    def minArea(self, area: float) -> "ImgDetectionsFilter":
        self._cfg.min_area = area
        return self

    def sortByConfidence(self, *, desc: bool = True) -> "ImgDetectionsFilter":
        """Enable sorting by confidence (before top-k). Set direction via `desc`."""
        self._cfg.sort_enabled = True
        self._cfg.sort_desc = desc
        return self

    def enableSorting(self) -> "ImgDetectionsFilter":
        """Enable sorting using the last configured sort settings."""
        self._cfg.sort_enabled = True
        return self

    def disableSorting(self) -> "ImgDetectionsFilter":
        """Disable sorting but keep the last configured sort settings."""
        self._cfg.sort_enabled = False
        return self

    def takeFirstK(self, k: Optional[int]):
        self._cfg.first_k = k
        return self

    def build(self, msg: dai.Node.Output) -> "ImgDetectionsFilter":
        self.link_args(msg)
        self._logger.debug(self._plan_string())
        return self

    def process(self, msg: dai.Buffer) -> None:
        assert isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections))
        msg_new = copy_message(msg)
        assert isinstance(msg_new, (dai.ImgDetections, dai.SpatialImgDetections))

        filtered_detections, filtered_out_ixs = self._filter_step(detections=msg.detections)
        sorted_detections = self._sorting_step(detections=filtered_detections)
        # Take first K step
        msg_new.detections = sorted_detections[: self._cfg.first_k]

        # Remove classes of filtered out detections
        if isinstance(msg, dai.ImgDetections):
            msg_new = self._update_segmentation_mask(msg_new=msg_new, filtered_out_ixs=filtered_out_ixs)

        self.out.send(msg_new)

    def _plan_string(self) -> str:
        return f"ImgDetectionsFilter plan: filter -> sort({self._cfg.sort}) -> take_first_k({self._cfg.first_k})"

    def _filter_step(
            self,
            detections: List[dai.ImgDetection | dai.SpatialImgDetection],
    ) -> Tuple[List[dai.ImgDetection | dai.SpatialImgDetection], List[int]]:
        filtered_detections = []
        filtered_out_ixs: List[int] = []
        for ix, detection in enumerate(detections):
            if self._cfg.labels_to_keep is not None:
                if detection.label not in self._cfg.labels_to_keep:
                    filtered_out_ixs.append(ix)
                    continue

            elif self._cfg.labels_to_reject is not None:
                if detection.label in self._cfg.labels_to_reject:
                    filtered_out_ixs.append(ix)
                    continue

            if self._cfg.min_confidence is not None:
                if detection.confidence < self._cfg.min_confidence:
                    filtered_out_ixs.append(ix)
                    continue

            if self._cfg.min_area is not None:
                area = compute_area(detection)
                if area < self._cfg.min_area:
                    filtered_out_ixs.append(ix)
                    continue

            filtered_detections.append(detection)
        return filtered_detections, filtered_out_ixs

    def _sorting_step(self, detections: List[dai.ImgDetection | dai.SpatialImgDetection]) -> List[dai.ImgDetection | dai.SpatialImgDetection]:
        if self._cfg.sort is not "none":
            reverse = self._cfg.sort == "confidence_desc"
            sorted_detections = sorted(
                detections, key=lambda x: x.confidence, reverse=reverse
            )
            return sorted_detections
        return detections

    @staticmethod
    def _update_segmentation_mask(msg_new, filtered_out_ixs: List[int]):
        mask = msg_new.getCvSegmentationMask()
        if mask is not None:
            mask = np.where(np.isin(mask, filtered_out_ixs), 255, mask)
            msg_new.setCvSegmentationMask(mask)
        return msg_new
