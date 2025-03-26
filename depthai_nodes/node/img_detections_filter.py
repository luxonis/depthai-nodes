import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from typing import List


class ImgDetectionsFilter(dai.node.HostNode):
    """
    Filters out detections that do not meet the specified criteria.

    Attributes
    ----------
    labels_to_keep : List[int]
        Labels to keep. Only detections with labels in this list will be kept.
    confidence_threshold : float
        Minimum confidence threshold. Detections with confidence below this threshold will be filtered out.
    max_detections : int
        Maximum number of detections to keep. If not defined, all detections will be kept.
    """

    def __init__(self):
        super().__init__()

        self.output = self.createOutput()

    def build(
        self,
        msg: dai.Node.Output,
        labels_to_keep: List[int] = None,
        confidence_threshold: float = None,
        max_detections: int = None,
    ) -> "ImgDetectionsFilter":

        self.link_args(msg)

        self._labels_to_keep = labels_to_keep
        self._confidence_threshold = confidence_threshold
        self._max_detections = max_detections

        return self

    def process(self, msg: dai.Buffer) -> None:
        assert isinstance(
            msg,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
            ),  # TODO: also allow dai.SpatialImgDetections?
        )

        filtered_detections = []
        for detection in msg.detections:

            if self._labels_to_keep is not None:
                if detection.label not in self._labels_to_keep:
                    continue

            if self._confidence_threshold is not None:
                if detection.confidence < self._confidence_threshold:
                    continue

            filtered_detections.append(detection)

        # TODO: sort detections by confidence before subsetting?
        msg.detections = filtered_detections[: self._max_detections]

        self.output.send(msg)
