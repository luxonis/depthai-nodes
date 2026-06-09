import copy

import depthai as dai

from depthai_nodes import KEYPOINT_COLOR, PRIMARY_COLOR
from depthai_nodes.utils import AnnotationHelper


class Keypoints(dai.Buffer):
    """DepthAI Nodes keypoints message wrapping a native ``dai.KeypointsList``."""

    def __init__(self):
        super().__init__()
        self._keypoints_list = dai.KeypointsList()
        self._transformation: dai.ImgTransformation | None = None

    def copy(self):
        new_obj = Keypoints()
        native_copy = dai.KeypointsList()
        native_copy.setKeypoints([copy.deepcopy(kp) for kp in self.getKeypoints()])
        native_copy.setEdges(copy.deepcopy(self.getEdges()))
        new_obj.keypoints_list = native_copy
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.getTransformation())
        return new_obj

    @property
    def keypoints_list(self) -> dai.KeypointsList:
        return self._keypoints_list

    @keypoints_list.setter
    def keypoints_list(self, value: dai.KeypointsList):
        if not isinstance(value, dai.KeypointsList):
            raise TypeError(
                f"keypoints_list must be a dai.KeypointsList, got {type(value)}."
            )
        self._keypoints_list = value

    @property
    def transformation(self) -> dai.ImgTransformation | None:
        return self._transformation

    @transformation.setter
    def transformation(self, value: dai.ImgTransformation | None):
        if value is not None and not isinstance(value, dai.ImgTransformation):
            raise TypeError(
                f"Transformation must be a dai.ImgTransformation object, got {type(value)}."
            )
        self._transformation = value

    def setTransformation(self, transformation: dai.ImgTransformation | None):
        self.transformation = transformation

    def getTransformation(self) -> dai.ImgTransformation | None:
        return self.transformation

    def getKeypoints(self) -> list[dai.Keypoint]:
        return self._keypoints_list.getKeypoints()

    def setKeypoints(self, value: list[dai.Keypoint]):
        self._keypoints_list.setKeypoints(value)

    def getEdges(self) -> list[tuple[int, int]]:
        return self._keypoints_list.getEdges()

    def setEdges(self, value: list[tuple[int, int]]):
        self._keypoints_list.setEdges(value)

    def getPoints2f(self) -> dai.VectorPoint2f:
        return dai.VectorPoint2f(
            [
                dai.Point2f(kp.imageCoordinates.x, kp.imageCoordinates.y)
                for kp in self.getKeypoints()
            ]
        )

    def getPoints3f(self) -> list[dai.Point3f]:
        return [
            dai.Point3f(
                kp.imageCoordinates.x, kp.imageCoordinates.y, kp.imageCoordinates.z
            )
            for kp in self.getKeypoints()
        ]

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        annotation_helper = AnnotationHelper()
        annotation_helper.draw_points(
            points=self.getPoints2f(), color=KEYPOINT_COLOR, thickness=1
        )
        for edge in self.getEdges():
            pt1_ix, pt2_ix = edge
            pt1 = self.getKeypoints()[pt1_ix]
            pt2 = self.getKeypoints()[pt2_ix]
            annotation_helper.draw_line(
                pt1=(pt1.imageCoordinates.x, pt1.imageCoordinates.y),
                pt2=(pt2.imageCoordinates.x, pt2.imageCoordinates.y),
                color=PRIMARY_COLOR,
                thickness=1,
            )
        return annotation_helper.build(
            timestamp=self.getTimestamp(), sequence_num=self.getSequenceNum()
        )
