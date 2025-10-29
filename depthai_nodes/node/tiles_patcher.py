import datetime
from typing import Optional, Union

import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.message.clusters import Clusters
from depthai_nodes.message.img_detections import (
    ImgDetectionsExtended,
)
from depthai_nodes.message.keypoints import Keypoints
from depthai_nodes.message.lines import Lines
from depthai_nodes.message.map import Map2D
from depthai_nodes.message.prediction import Predictions
from depthai_nodes.message.segmentation import SegmentationMask
from depthai_nodes.node.utils import nms_detections
from depthai_nodes.node.utils.detection_merging import merge_messages
from depthai_nodes.node.utils.detection_remapping import remap_message


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

    SUPPORTED_MESSAGES = (
        dai.ImgDetections,
        ImgDetectionsExtended,
        Keypoints,
        SegmentationMask,
        Clusters,
        Map2D,
        Lines,
        Predictions,
    )

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

            nn_msgs = []
            if (
                last_nn_msg is not None
                and last_nn_msg.getTimestamp() == img.getTimestamp()
            ):
                nn_msgs.append(last_nn_msg)
                last_nn_msg = None
            while True:
                nn_msg = self._nn_input.get()
                assert isinstance(
                    nn_msg, self.SUPPORTED_MESSAGES
                ), f"Message type {type(nn_msg)} is not supported."
                if nn_msg.getTimestamp() > img.getTimestamp():
                    last_nn_msg = nn_msg
                    break
                nn_msgs.append(nn_msg)

            remapped_messages = [
                remap_message(
                    nn_msg.getTransformation(),  # type: ignore
                    img.getTransformation(),
                    nn_msg,
                )
                for nn_msg in nn_msgs
            ]
            merged_detections = merge_messages(remapped_messages)

            self._sendOutput(
                merged_detections,
                img.getTimestamp(),
                img.getTimestampDevice(),
                img.getTransformation(),
                img.getSequenceNum(),
            )

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        self.conf_thresh = confidence_threshold

    def _sendOutput(
        self,
        merged_detections: Union[
            dai.ImgDetections,
            ImgDetectionsExtended,
            Keypoints,
            SegmentationMask,
            Clusters,
            Map2D,
            Lines,
            Predictions,
        ],
        timestamp: datetime.timedelta,
        device_timestamp: datetime.timedelta,
        transformation: Optional[dai.ImgTransformation],
        sequence_num: int,
    ) -> None:
        """Send the final combined bounding boxes as output when all tiles for a frame
        are processed.

        @param timestamp: The timestamp of the frame.
        @param device_timestamp: The timestamp of the frame on the device.
        """
        if isinstance(merged_detections, (dai.ImgDetections, ImgDetectionsExtended)):
            merged_detections.detections = nms_detections(
                merged_detections.detections,  # type: ignore
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
            )
        merged_detections.setTimestamp(timestamp)
        merged_detections.setTimestampDevice(device_timestamp)
        merged_detections.setSequenceNum(sequence_num)
        if transformation is not None:
            merged_detections.setTransformation(transformation)

        self._logger.debug("Detections message created")

        self.out.send(merged_detections)

        self._logger.debug("Message sent successfully")
