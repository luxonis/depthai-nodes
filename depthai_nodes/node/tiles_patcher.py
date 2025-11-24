import datetime
from typing import List, Optional, Type, Union

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.logging import get_logger
from depthai_nodes.message.img_detections import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoint, Keypoints
from depthai_nodes.node.utils import nms_detections

UNASSIGNED_MASK_LABEL = -1


class TilesPatcher(dai.node.ThreadedHostNode):
    """Handles the processing of tiled frames from neural network (NN) outputs, maps the
    detections from tiles back into the global frame, and sends out the combined
    detections for further processing.

    @ivar conf_thresh: Confidence threshold for filtering detections.
    @type conf_thresh: float
    @ivar iou_thresh: IOU threshold for non-max suppression.
    @type iou_thresh: float
    @ivar tile_buffer: Buffer to store tile detections temporarily.
    @type tile_buffer: list
    @ivar current_timestamp: Timestamp for the current frame being processed.
    @type current_timestamp: float
    @ivar expected_tiles_count: Number of tiles expected per frame.
    @type expected_tiles_count: int
    """

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
        """Initializes the TilesPatcher node, sets default thresholds for confidence and
        IOU, and initializes buffers for tile processing."""
        super().__init__()
        self._pipeline = self.getParentPipeline()
        platform = self._pipeline.getDefaultDevice().getPlatform()
        if platform == dai.Platform.RVC2:
            raise RuntimeError("TilesPatcher node is currently not supported on RVC2.")
        self._logger = get_logger(self.__class__.__name__)
        self.name = "TilesPatcher"
        self.conf_thresh = 0.3
        self.iou_thresh = 0.4

        self._nn_input = self.createInput()
        self._img_input = self.createInput()
        self.out = self.createOutput()
        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)
        self._logger.debug("TilesPatcher initialized")

    def build(
        self,
        img_frames: dai.Node.Output,
        nn: dai.Node.Output,
        conf_thresh=0.3,
        iou_thresh=0.4,
    ) -> "TilesPatcher":
        """Configures the TilesPatcher node with the tile manager and links the neural
        network's output.

        @param nn: The output of the neural network node from which detections are
            received.
        @type nn: dai.Node.Output
        @param conf_thresh: Confidence threshold for filtering detections (default:
            0.3).
        @type conf_thresh: float
        @param iou_thresh: IOU threshold for non-max suppression (default: 0.4).
        @type iou_thresh: float
        @return: Returns self for method chaining.
        @rtype: TilesPatcher
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        img_frames.link(self._script.inputs["preview"])
        self._script.outputs["transformation"].link(self._img_input)
        nn.link(self._nn_input)
        return self

    def run(self):
        last_nn_msg = None
        while self.isRunning():
            img = self._img_input.get()
            assert isinstance(img, dai.ImgFrame)

            nn_msgs: List[Union[dai.ImgDetections, ImgDetectionsExtended]] = []
            if (
                last_nn_msg is not None
                and last_nn_msg.getTimestamp() == img.getTimestamp()
            ):
                nn_msgs.append(last_nn_msg)
                last_nn_msg = None
            while True:
                nn_msg = self._nn_input.get()
                assert isinstance(nn_msg, (dai.ImgDetections, ImgDetectionsExtended))
                if nn_msg.getTimestamp() != img.getTimestamp():
                    last_nn_msg = nn_msg
                    break
                nn_msgs.append(nn_msg)

            remapped_detections: List[
                Union[dai.ImgDetection, ImgDetectionExtended]
            ] = []
            for nn_msg in nn_msgs:
                mapped_dets = self._remapDetections(
                    nn_msg.getTransformation(),  # type: ignore
                    img.getTransformation(),
                    nn_msg.detections,
                )
                remapped_detections.extend(mapped_dets)

            if all(isinstance(nn_msg, ImgDetectionsExtended) for nn_msg in nn_msgs):
                remapped_seg_maps = []
                for nn_msg in nn_msgs:
                    if nn_msg.masks.size == 0:  # type: ignore
                        continue
                    remapped_seg_map = self._remapSegmentationMap(
                        nn_msg.getTransformation(),  # type: ignore
                        img.getTransformation(),
                        nn_msg.masks,  # type: ignore
                    )
                    remapped_seg_maps.append(remapped_seg_map)
                if len(remapped_seg_maps) == 0:
                    full_seg_map = np.empty(0, dtype=np.int16)
                else:
                    full_seg_map = self._stitchSegmentationMaps(remapped_seg_maps)
            else:
                full_seg_map = None

            if len(nn_msgs) > 0:
                self._sendOutput(
                    remapped_detections,
                    full_seg_map,
                    img.getTimestamp(),
                    img.getTimestampDevice(),
                    img.getTransformation(),
                    img.getSequenceNum(),
                    type(nn_msgs[0]),
                )

    def _stitchSegmentationMaps(self, segmentation_maps: List[np.ndarray]):
        full_seg_map = np.full_like(segmentation_maps[0], UNASSIGNED_MASK_LABEL)
        for seg_map in segmentation_maps:
            full_seg_map[seg_map != UNASSIGNED_MASK_LABEL] = seg_map[
                seg_map != UNASSIGNED_MASK_LABEL
            ]
        return full_seg_map

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

    def _sendOutput(
        self,
        remapped_detections: List[Union[dai.ImgDetection, ImgDetectionExtended]],
        seg_map: Optional[np.ndarray],
        timestamp: datetime.timedelta,
        device_timestamp: datetime.timedelta,
        transformation: Optional[dai.ImgTransformation],
        sequence_num: int,
        input_type: Type[Union[dai.ImgDetections, ImgDetectionsExtended]],
    ) -> None:
        """Send the final combined bounding boxes as output when all tiles for a frame
        are processed.

        @param timestamp: The timestamp of the frame.
        @param device_timestamp: The timestamp of the frame on the device.
        """
        detection_list = nms_detections(
            remapped_detections,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
        )
        if input_type == dai.ImgDetections:
            detections = dai.ImgDetections()
            detections.detections = detection_list
        elif input_type == ImgDetectionsExtended:
            detections = ImgDetectionsExtended()
            if seg_map is not None:
                detections.masks = seg_map
            detections.detections = detection_list
        else:
            raise ValueError("Unsupported input type")

        detections.setTimestamp(timestamp)
        detections.setTimestampDevice(device_timestamp)
        detections.setSequenceNum(sequence_num)
        if transformation is not None:
            detections.setTransformation(transformation)

        self._logger.debug("Detections message created")

        self.out.send(detections)

        self._logger.debug("Message sent successfully")
