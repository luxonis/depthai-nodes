import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.utils import compute_area, copy_message
from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.utils.nms import nms_detections


@dataclass
class _FilterCfg:
    labels_to_keep: Optional[List[int]] = None
    labels_to_reject: Optional[List[int]] = None
    min_confidence: Optional[float] = None
    min_area: Optional[float] = None
    nms_disabled: bool = True
    nms_conf_thresh: float = 0.3
    nms_iou_thresh: float = 0.4
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
    keepLabels(labels)
        Keep only detections whose label is present in ``labels``.
    rejectLabels(labels)
        Drop detections whose label is present in ``labels``.
    minConfidence(threshold)
        Require detections to meet a minimum confidence.
    minArea(area)
        Require detections to meet a minimum normalized bounding-box area.
    useNms(confThresh=..., iouThresh=...)
        Enable non-maximum suppression after filtering.
    sortByConfidence(desc=True)
        Sort detections by confidence before optional top-k truncation.
    takeFirstK(k)
        Keep only the first ``k`` detections after all previous steps.
    """

    def __init__(self):
        super().__init__()
        self._cfg = _FilterCfg()
        self._logger.debug("ImgDetectionsFilter initialized")

    def setLabels(self, labels: List[int], keep: bool) -> None:
        """Deprecated wrapper for configuring label inclusion or exclusion."""
        warnings.warn(
            "setLabels() is deprecated; use keepLabels() or rejectLabels() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if keep:
            self.keepLabels(labels=labels)
        else:
            self.rejectLabels(labels=labels)

    def setConfidenceThreshold(self, confidenceThreshold: float | None) -> None:
        """Deprecated wrapper for setting the minimum confidence."""
        warnings.warn(
            "setConfidenceThreshold() is deprecated; use minConfidence() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minConfidence(threshold=confidenceThreshold)

    def setMaxDetections(self, maxDetections: int) -> None:
        """Deprecated wrapper for limiting the number of detections."""
        warnings.warn(
            "setMaxDetections() is deprecated; use takeFirstK() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.takeFirstK(k=maxDetections)

    def setSortByConfidence(self, sortByConfidence: bool) -> None:
        """Deprecated wrapper for toggling confidence-based sorting."""
        warnings.warn(
            "setSortByConfidence() is deprecated; use sortByConfidence(), enableSorting() and disableSorting() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if sortByConfidence is True:
            self.enableSorting()
        else:
            self.disableSorting()

    def setMinArea(self, minArea: float) -> None:
        """Deprecated wrapper for setting the minimum detection area."""
        warnings.warn(
            "setMinArea() is deprecated; use minArea() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minArea(area=minArea)

    def keepLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        """Keep only detections whose label is in ``labels``."""
        self._cfg.labels_to_keep = labels
        if self._cfg.labels_to_reject is not None:
            self._logger.warn(f"Removing labels to reject. Use either `keepLabels` or `rejectLabels` but not both.")
            self._cfg.labels_to_reject = None
        return self

    def rejectLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        """Drop detections whose label is in ``labels``."""
        self._cfg.labels_to_reject = labels
        if self._cfg.labels_to_keep is not None:
            self._logger.warn(f"Removing labels to keep. Use either `keepLabels` or `rejectLabels` but not both.")
            self._cfg.labels_to_keep = None
        return self

    def minConfidence(self, threshold: float) -> "ImgDetectionsFilter":
        """Require detections to meet the minimum confidence threshold."""
        self._cfg.min_confidence = threshold
        return self

    def minArea(self, area: float) -> "ImgDetectionsFilter":
        """Require detections to meet the minimum normalized bounding-box area."""
        self._cfg.min_area = area
        return self

    def sortByConfidence(self, *, desc: bool = True) -> "ImgDetectionsFilter":
        """Enable sorting by confidence (before top-k). Set direction via `desc`."""
        self._cfg.sort_disabled = False
        self._cfg.sort_desc = desc
        return self

    def useNms(
        self, *, confThresh: float = 0.3, iouThresh: float = 0.4
    ) -> "ImgDetectionsFilter":
        """Enable NMS after filtering and configure its thresholds."""
        self._cfg.nms_disabled = False
        self._cfg.nms_conf_thresh = confThresh
        self._cfg.nms_iou_thresh = iouThresh
        return self

    def enableSorting(self) -> "ImgDetectionsFilter":
        """Enable sorting using the last configured sort settings."""
        self._cfg.sort_disabled = False
        return self

    def disableSorting(self) -> "ImgDetectionsFilter":
        """Disable sorting but keep the last configured sort settings."""
        self._cfg.sort_disabled = True
        return self

    def takeFirstK(self, k: Optional[int]):
        """Keep only the first ``k`` detections after filtering and sorting."""
        self._cfg.first_k = k
        return self

    def build(self, input: dai.Node.Output) -> "ImgDetectionsFilter":
        """Connect the detections stream to the filter node."""
        self.link_args(input)
        self._logger.debug(self._plan_string())
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Filter, optionally suppress, sort, and emit the detections message."""
        assert isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections))
        msg_new = copy_message(msg)
        assert isinstance(msg_new, (dai.ImgDetections, dai.SpatialImgDetections))

        filtered_detections, filtered_out_ixs = self._filter_step(detections=msg.detections)
        nms_detections_out = self._nms_step(detections=filtered_detections)
        sorted_detections = self._sorting_step(detections=nms_detections_out)
        # Take first K step
        msg_new.detections = sorted_detections[: self._cfg.first_k]

        # Remove classes of filtered out detections
        if isinstance(msg, dai.ImgDetections):
            msg_new = self._update_segmentation_mask(msg_new=msg_new, filtered_out_ixs=filtered_out_ixs)

        self.out.send(msg_new)

    def _plan_string(self) -> str:
        return (f"ImgDetectionsFilter plan: filter -> nms(enabled: {not self._cfg.nms_disabled}, confidence threshold: {self._cfg.nms_conf_thresh}, iou threshold: {self._cfg.nms_iou_thresh})"
                f" -> sort(enabled: {not self._cfg.sort_disabled}, descending order: {self._cfg.sort_desc})"
                f" -> take_first_k({self._cfg.first_k})")

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

    def _nms_step(
        self, detections: List[dai.ImgDetection | dai.SpatialImgDetection]
    ) -> List[dai.ImgDetection | dai.SpatialImgDetection]:
        if self._cfg.nms_disabled:
            return detections
        return nms_detections(
            detections=detections,
            conf_thresh=self._cfg.nms_conf_thresh,
            iou_thresh=self._cfg.nms_iou_thresh,
        )

    def _sorting_step(self, detections: List[dai.ImgDetection | dai.SpatialImgDetection]) -> List[dai.ImgDetection | dai.SpatialImgDetection]:
        if not self._cfg.sort_disabled:
            sorted_detections = sorted(
                detections, key=lambda x: x.confidence, reverse=self._cfg.sort_desc
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
