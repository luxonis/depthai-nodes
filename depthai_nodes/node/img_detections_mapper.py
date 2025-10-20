from typing import List, Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoint, Keypoints
from depthai_nodes.node.base_host_node import BaseHostNode

UNASSIGNED_MASK_LABEL = -1


class ImgDetectionsMapper(BaseHostNode):
    """Remap ImgDetections to ImgFrame coordinates."""

    SCRIPT_CONTENT = """
# Strip ImgFrame image data and send only ImgTransformation
# Reduces the amount of date being sent between host and device

try:
    while True:
        frame = node.inputs['preview'].get()
        transformation = frame.getTransformation()
        empty_frame = ImgFrame()
        empty_frame.setTransformation(transformation)
        empty_frame.setTimestamp(frame.getTimestamp())
        empty_frame.setTimestampDevice(frame.getTimestampDevice())
        node.outputs['transformation'].send(empty_frame)
except Exception as e:
    node.warn(str(e))
"""

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)
        self._logger.debug("ImgDetectionsMapper initialized")

    def build(
        self, img_input: dai.Node.Output, nn_input: dai.Node.Output
    ) -> "ImgDetectionsMapper":
        img_input.link(self._script.inputs["preview"])
        self._script.outputs["transformation"].setPossibleDatatypes(
            [(dai.DatatypeEnum.ImgFrame, True)]
        )
        self.link_args(self._script.outputs["transformation"], nn_input)
        return self

    def process(self, img: dai.ImgFrame, nn: dai.Buffer) -> None:
        assert isinstance(
            nn, (dai.ImgDetections, ImgDetectionsExtended)
        ), "Expected ImgDetections or ImgDetectionsExtended"

        nn_trans = nn.getTransformation()
        if nn_trans is None:
            raise RuntimeError("Received detection message without transformation")
        remapped_detections = self._remapDetections(
            nn_trans,
            img.getTransformation(),
            nn.detections,
        )
        if isinstance(nn, ImgDetectionsExtended):
            if nn.masks.size == 0:
                remapped_seg_maps = np.empty(0, dtype=np.int16)
            else:
                remapped_seg_maps = self._remapSegmentationMap(
                    nn_trans,
                    img.getTransformation(),
                    nn.masks,
                )
            message = ImgDetectionsExtended()
            message.detections = remapped_detections
            message.masks = remapped_seg_maps
        else:
            message = dai.ImgDetections()
            message.detections = remapped_detections
        message.setTimestamp(nn.getTimestamp())
        message.setTimestampDevice(nn.getTimestampDevice())
        message.setSequenceNum(nn.getSequenceNum())
        message.setTransformation(img.getTransformation())
        self.out.send(message)

    def _remapSegmentationMap(
        self,
        src_transformation: dai.ImgTransformation,
        dst_transformation: dai.ImgTransformation,
        seg_map: np.ndarray,
    ):
        dst_matrix = np.array(dst_transformation.getMatrix())
        src_matrix = np.array(src_transformation.getMatrixInv())
        trans_matrix = dst_matrix @ src_matrix
        res = cv2.warpPerspective(
            seg_map,
            trans_matrix,
            dst_transformation.getSize(),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=UNASSIGNED_MASK_LABEL,  # type: ignore
        )
        return res

    def _remapDetection(
        self,
        src_transformation: dai.ImgTransformation,
        dst_transformation: dai.ImgTransformation,
        detection: dai.ImgDetection,
    ):
        new_det = dai.ImgDetection()
        min_pt = src_transformation.remapPointTo(
            dst_transformation,
            dai.Point2f(np.clip(detection.xmin, 0, 1), np.clip(detection.ymin, 0, 1)),
        )
        max_pt = src_transformation.remapPointTo(
            dst_transformation,
            dai.Point2f(np.clip(detection.xmax, 0, 1), np.clip(detection.ymax, 0, 1)),
        )
        new_det.xmin = max(0, min(min_pt.x, 1))
        new_det.ymin = max(0, min(min_pt.y, 1))
        new_det.xmax = max(0, min(max_pt.x, 1))
        new_det.ymax = max(0, min(max_pt.y, 1))
        new_det.label = detection.label
        new_det.confidence = detection.confidence
        return new_det

    def _remapDetectionExtended(
        self,
        src_transformation: dai.ImgTransformation,
        dst_transformation: dai.ImgTransformation,
        detection: ImgDetectionExtended,
    ):
        new_det = ImgDetectionExtended()

        if detection.rotated_rect.angle == 0:
            new_rect = src_transformation.remapRectTo(
                dst_transformation, detection.rotated_rect
            )
            new_det.rotated_rect = (
                new_rect.center.x,
                new_rect.center.y,
                new_rect.size.width,
                new_rect.size.height,
                new_rect.angle,
            )
        else:
            # TODO: This is a temporary fix - DepthAI doesn't handle rotated rects with angle != 0 correctly
            pts = detection.rotated_rect.getPoints()
            pts = [dai.Point2f(np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in pts]
            remapped_pts = [
                src_transformation.remapPointTo(dst_transformation, pt) for pt in pts
            ]
            remapped_pts = [
                (np.clip(pt.x, 0, 1), np.clip(pt.y, 0, 1)) for pt in remapped_pts
            ]
            (center_x, center_y), (width, height), angle = cv2.minAreaRect(
                np.array(remapped_pts, dtype=np.float32)
            )
            new_det.rotated_rect = (
                center_x,
                center_y,
                width,
                height,
                angle,
            )
        new_det.confidence = detection.confidence
        new_det.label = detection.label
        new_det.label_name = detection.label_name

        new_kpts_list = []
        for kpt in detection.keypoints:
            remapped_kpt = src_transformation.remapPointTo(
                dst_transformation, dai.Point2f(kpt.x, kpt.y)
            )
            new_kpt = Keypoint()
            new_kpt.x = remapped_kpt.x
            new_kpt.y = remapped_kpt.y
            new_kpt.z = kpt.z
            new_kpts_list.append(new_kpt)
        new_kpts = Keypoints()
        new_kpts.keypoints = new_kpts_list
        new_det.keypoints = new_kpts
        return new_det

    def _remapDetections(
        self,
        src_transformation: dai.ImgTransformation,
        dst_transformation: dai.ImgTransformation,
        detections: Union[List[dai.ImgDetection], List[ImgDetectionExtended]],
    ) -> Union[List[dai.ImgDetection], List[ImgDetectionExtended]]:
        new_dets = []
        for det in detections:
            if isinstance(det, dai.ImgDetection):
                new_det = self._remapDetection(
                    src_transformation, dst_transformation, det
                )
            elif isinstance(det, ImgDetectionExtended):
                new_det = self._remapDetectionExtended(
                    src_transformation, dst_transformation, det
                )
            new_dets.append(new_det)
        return new_dets
