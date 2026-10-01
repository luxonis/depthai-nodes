import warnings
from dataclasses import dataclass

import depthai as dai
import numpy as np

from depthai_nodes.message.utils import compute_area, copy_message
from depthai_nodes.node.base_host_node import BaseHostNode
from depthai_nodes.node.utils.nms import nms_detections


@dataclass
class _FilterCfg:
    labels_to_keep: list[int] | None = None
    labels_to_reject: list[int] | None = None
    min_confidence: float | None = None
    min_area: float | None = None
    nms_disabled: bool = True
    nms_conf_thresh: float = 0.3
    nms_iou_thresh: float = 0.4
    sort_desc: bool = True
    sort_disabled: bool = True
    first_k: int | None = None


class ImgDetectionsFilter(BaseHostNode):
    """Filter detections and send the retained items as a separate message.

    Filtering runs in this order: label inclusion or exclusion, confidence, area,
    optional non-maximum suppression, optional confidence sorting, and optional
    truncation to the first K detections.

    Configure these stages with ``keepLabels()``, ``rejectLabels()``,
    ``minConfidence()``, ``minArea()``, ``useNms()``, ``sortByConfidence()``,
    and ``takeFirstK()``. Sorting and non-maximum suppression are disabled
    by default.

    Attributes:
        out: Output stream of ``dai.ImgDetections`` or ``dai.SpatialImgDetections``
            messages.
    """

    def __init__(self):
        super().__init__()
        self._cfg = _FilterCfg()
        self.out.setPossibleDatatypes(
            [
                (dai.DatatypeEnum.ImgDetections, True),
                (dai.DatatypeEnum.SpatialImgDetections, True),
            ]
        )
        self._logger.debug("ImgDetectionsFilter initialized")

    def setLabels(self, labels: list[int], keep: bool) -> None:
        """Configure filtering through a deprecated compatibility setter.

        Args:
            labels: Class indexes to include or exclude.
            keep: Include the labels when true; exclude them otherwise.

        Note:
            Emits ``FutureWarning``. Use ``keepLabels() or rejectLabels()`` instead.
        """
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
        """Configure filtering through a deprecated compatibility setter.

        Args:
            confidenceThreshold: Minimum confidence, or None to disable filtering.

        Note:
            Emits ``FutureWarning``. Use ``minConfidence()`` instead.
        """
        warnings.warn(
            "setConfidenceThreshold() is deprecated; use minConfidence() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minConfidence(threshold=confidenceThreshold)

    def setMaxDetections(self, maxDetections: int) -> None:
        """Configure filtering through a deprecated compatibility setter.

        Args:
            maxDetections: Slice stop index for the retained detections.

        Note:
            Emits ``FutureWarning``. Use ``takeFirstK()`` instead.
        """
        warnings.warn(
            "setMaxDetections() is deprecated; use takeFirstK() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.takeFirstK(k=maxDetections)

    def setSortByConfidence(self, sortByConfidence: bool) -> None:
        """Configure filtering through a deprecated compatibility setter.

        Args:
            sortByConfidence: Enable sorting when true; disable it otherwise.

        Note:
            Emits ``FutureWarning``. Use ``enableSorting() or disableSorting()``
            instead.
        """
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
        """Configure filtering through a deprecated compatibility setter.

        Args:
            minArea: Minimum normalized bounding-box area.

        Note:
            Emits ``FutureWarning``. Use ``minArea()`` instead.
        """
        warnings.warn(
            "setMinArea() is deprecated; use minArea() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.minArea(area=minArea)

    def keepLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        """Keep only the selected labels and clear any rejection list.

        Args:
            labels: Class indexes to retain. An empty list rejects all detections.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.labels_to_keep = labels
        if self._cfg.labels_to_reject is not None:
            self._logger.warn(
                "Removing labels to reject. Use either `keepLabels` or `rejectLabels` but not both."
            )
            self._cfg.labels_to_reject = None
        return self

    def rejectLabels(self, labels: list[int]) -> "ImgDetectionsFilter":
        """Reject selected labels and clear any inclusion list.

        Args:
            labels: Class indexes to remove. An empty list removes no detections.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.labels_to_reject = labels
        if self._cfg.labels_to_keep is not None:
            self._logger.warn(
                "Removing labels to keep. Use either `keepLabels` or `rejectLabels` but not both."
            )
            self._cfg.labels_to_keep = None
        return self

    def minConfidence(self, threshold: float) -> "ImgDetectionsFilter":
        """Configure the inclusive minimum detection confidence.

        Args:
            threshold: Minimum score; ``None`` disables this filter.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.min_confidence = threshold
        return self

    def minArea(self, area: float) -> "ImgDetectionsFilter":
        """Configure the inclusive minimum normalized bounding-box area.

        Args:
            area: Minimum width-times-height area; ``None`` disables this filter.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.min_area = area
        return self

    def sortByConfidence(self, *, desc: bool = True) -> "ImgDetectionsFilter":
        """Enable confidence sorting before the count limit.

        Args:
            desc: Sort highest confidence first when true; lowest first otherwise.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.sort_disabled = False
        self._cfg.sort_desc = desc
        return self

    def useNms(
        self, *, confThresh: float = 0.3, iouThresh: float = 0.4
    ) -> "ImgDetectionsFilter":
        """Enable per-class suppression after label, confidence, and area filtering.

        Args:
            confThresh: Minimum confidence passed to suppression.
            iouThresh: Intersection-over-union threshold for overlapping boxes.

        Returns:
            This node for fluent configuration.
        """
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

    def takeFirstK(self, k: int | None):
        """Configure slicing after filtering, suppression, and sorting.

        Args:
            k: Slice stop index. ``None`` retains all detections, zero retains none, and
                negative values follow Python slice semantics.

        Returns:
            This node for fluent configuration.
        """
        self._cfg.first_k = k
        return self

    def build(self, input: dai.Node.Output) -> "ImgDetectionsFilter":
        """Connect the stream of detections to filter.

        Args:
            input: Output producing ``dai.ImgDetections`` or
                ``dai.SpatialImgDetections``.

        Returns:
            This node for fluent configuration.
        """
        self.link_args(input)
        self._logger.debug(self._plan_string())
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Filter a copied message and emit it through ``out``.

        Args:
            msg: Native image or spatial detections. Attached uint8 instance-mask IDs
                are reindexed to match retained detections; removed IDs become
                background 255.

        Raises:
            AssertionError: If the input or its copy is not a supported detection
                message.
        """
        assert isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections))
        msg_new = copy_message(msg)
        assert isinstance(msg_new, (dai.ImgDetections, dai.SpatialImgDetections))

        indexed_detections = list(enumerate(msg.detections))
        filtered_detections = self._filter_step(detections=indexed_detections)
        nms_detections_out = self._nms_step(detections=filtered_detections)
        sorted_detections = self._sorting_step(detections=nms_detections_out)
        # Take first K step
        final_detections = sorted_detections[: self._cfg.first_k]
        msg_new.detections = [detection for _, detection in final_detections]

        # Reindex segmentation mask instance IDs to match final detections.
        if isinstance(msg, dai.ImgDetections):
            msg_new = self._update_segmentation_mask(
                msg_new=msg_new,
                kept_original_indices=[ix for ix, _ in final_detections],
            )

        self.out.send(msg_new)

    def _plan_string(self) -> str:
        return (
            f"ImgDetectionsFilter plan: filter -> nms(enabled: {not self._cfg.nms_disabled}, confidence threshold: {self._cfg.nms_conf_thresh}, iou threshold: {self._cfg.nms_iou_thresh})"
            f" -> sort(enabled: {not self._cfg.sort_disabled}, descending order: {self._cfg.sort_desc})"
            f" -> take_first_k({self._cfg.first_k})"
        )

    def _filter_step(
        self,
        detections: list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]],
    ) -> list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]]:
        filtered_detections = []
        for ix, detection in detections:
            if self._cfg.labels_to_keep is not None:
                if detection.label not in self._cfg.labels_to_keep:
                    continue

            elif self._cfg.labels_to_reject is not None:
                if detection.label in self._cfg.labels_to_reject:
                    continue

            if self._cfg.min_confidence is not None:
                if detection.confidence < self._cfg.min_confidence:
                    continue

            if self._cfg.min_area is not None:
                area = compute_area(detection)
                if area < self._cfg.min_area:
                    continue

            filtered_detections.append((ix, detection))
        return filtered_detections

    def _nms_step(
        self, detections: list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]]
    ) -> list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]]:
        if self._cfg.nms_disabled:
            return detections
        detections_only = [detection for _, detection in detections]
        kept_detections = nms_detections(
            detections=detections_only,
            conf_thresh=self._cfg.nms_conf_thresh,
            iou_thresh=self._cfg.nms_iou_thresh,
        )
        by_id = {id(detection): (ix, detection) for ix, detection in detections}
        return [by_id[id(detection)] for detection in kept_detections]

    def _sorting_step(
        self, detections: list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]]
    ) -> list[tuple[int, dai.ImgDetection | dai.SpatialImgDetection]]:
        if not self._cfg.sort_disabled:
            sorted_detections = sorted(
                detections,
                key=lambda indexed_detection: indexed_detection[1].confidence,
                reverse=self._cfg.sort_desc,
            )
            return sorted_detections
        return detections

    @staticmethod
    def _update_segmentation_mask(msg_new, kept_original_indices: list[int]):
        mask = msg_new.getCvSegmentationMask()
        if mask is not None:
            lookup_table = np.full(256, 255, dtype=np.uint8)
            for new_ix, original_ix in enumerate(kept_original_indices):
                if original_ix < 255 and new_ix < 255:
                    lookup_table[original_ix] = new_ix
            mask = lookup_table[mask].astype(np.uint8, copy=False)
            msg_new.setCvSegmentationMask(mask)
        return msg_new
