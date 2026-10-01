from collections.abc import Callable
from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils import top_left_wh_to_xywh
from depthai_nodes.node.parsers.utils.nms import nms_cv2
from depthai_nodes.node.parsers.utils.yunet import (
    compute_yunet_detections,
    generate_anchors,
)


class YuNetParser(DetectionParser):
    """Parser class for parsing the output of the YuNet face detection model.

    Attributes:
        conf_threshold (``float``): Confidence score threshold for detected faces.
        iou_threshold (``float``): Non-maximum suppression threshold.
        max_det (``int``): Maximum number of detections to keep.
        input_size (``tuple[int, int]``): Input size (width, height).
        loc_output_layer_name (``str``): Name of the output layer containing the
            location predictions.
        conf_output_layer_name (``str``): Name of the output layer containing the
            confidence predictions.
        iou_output_layer_name (``str``): Name of the output layer containing the IoU
            predictions.

    Note:
        Emits ``dai.ImgDetections`` messages. dai.ImgDetections message containing
        bounding boxes, labels, confidence scores, and keypoints of detected faces.
    """

    def __init__(
        self,
        conf_threshold: float = 0.8,
        iou_threshold: float = 0.3,
        max_det: int = 5000,
        input_size: tuple[int, int] = None,
        loc_output_layer_name: str = None,
        conf_output_layer_name: str = None,
        iou_output_layer_name: str = None,
    ) -> None:
        """Initializes the parser node.

        Args:
            conf_threshold: Confidence score threshold for detected faces.
            iou_threshold: Non-maximum suppression threshold.
            max_det: Maximum number of detections to keep.
            input_size: Input size of the model (width, height).
            loc_output_layer_name: Output layer name for the location predictions.
            conf_output_layer_name: Output layer name for the confidence predictions.
            iou_output_layer_name: Output layer name for the IoU predictions.
        """
        super().__init__(conf_threshold, iou_threshold, max_det)
        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )
        self.loc_output_layer_name = loc_output_layer_name
        self.conf_output_layer_name = conf_output_layer_name
        self.iou_output_layer_name = iou_output_layer_name
        self.input_size = input_size
        self.label_names = ["Face"]
        self._logger.debug(
            f"YuNetParser initialized with conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}, max_det={self.max_det}"
        )

        # Cache for anchors to avoid regeneration
        self._cached_anchors = None
        self._cached_input_size = None

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        """Sets the input size of the model.

        Args:
            input_size: Input size of the model (width, height).
        """
        if not isinstance(input_size, tuple):
            raise ValueError("Input size must be a tuple.")
        if not all(isinstance(size, int) for size in input_size):
            raise ValueError("Input size must be a tuple of integers.")
        self.input_size = input_size
        self._logger.debug(
            f"Input size updated to (width={input_size[0]}, height={input_size[1]})"
        )

    def setOutputLayerLoc(self, loc_output_layer_name: str) -> None:
        """Sets the name of the output layer containing the location predictions.

        Args:
            loc_output_layer_name: Output layer name for the loc tensor.
        """
        if not isinstance(loc_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.loc_output_layer_name = loc_output_layer_name
        self._logger.debug(
            f"Location output layer name set to '{self.loc_output_layer_name}'"
        )

    def setOutputLayerConf(self, conf_output_layer_name: str) -> None:
        """Sets the name of the output layer containing the confidence predictions.

        Args:
            conf_output_layer_name: Output layer name for the conf tensor.
        """
        if not isinstance(conf_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.conf_output_layer_name = conf_output_layer_name
        self._logger.debug(
            f"Confidence output layer name set to '{self.conf_output_layer_name}'"
        )

    def setOutputLayerIou(self, iou_output_layer_name: str) -> None:
        """Sets the name of the output layer containing the IoU predictions.

        Args:
            iou_output_layer_name: Output layer name for the IoU tensor.
        """
        if not isinstance(iou_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.iou_output_layer_name = iou_output_layer_name
        self._logger.debug(
            f"IoU output layer name set to '{self.iou_output_layer_name}'"
        )

    def build(
        self,
        head_config: dict[str, Any],
    ) -> "YuNetParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        super().build(head_config)
        output_layers = head_config.get("outputs", [])
        for output_layer in output_layers:
            if "loc" in output_layer:
                self.loc_output_layer_name = output_layer
            elif "conf" in output_layer:
                self.conf_output_layer_name = output_layer
            elif "iou" in output_layer:
                self.iou_output_layer_name = output_layer
            else:
                raise ValueError(
                    f"Unexpected output layer {output_layer}. Only loc, conf, and iou output layers are supported."
                )
        inputs = head_config["model_inputs"]
        if len(inputs) != 1:
            raise ValueError(
                f"Only one input supported for YuNetParser, got {len(inputs)} inputs."
            )
        self.input_shape = inputs[0].get("shape")
        self.layout = inputs[0].get("layout")
        if self.layout == "NHWC":
            self.input_size = (self.input_shape[2], self.input_shape[1])
        elif self.layout == "NCHW":
            self.input_size = (self.input_shape[3], self.input_shape[2])
        else:
            raise ValueError(
                f"Input layout {self.layout} not supported for input_size extraction."
            )

        self._logger.debug(
            f"YuNetParser built with input_size={self.input_size}, layout='{self.layout}'"
        )
        return self

    def _get_cached_anchors(self):
        """Get cached anchors or generate new ones if input_size changed."""
        if self._cached_anchors is None or self._cached_input_size != self.input_size:
            self._cached_anchors = generate_anchors(self.input_size)
            self._cached_input_size = self.input_size
        return self._cached_anchors

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("YuNetParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            loc, conf, iou = self.extract(output)
            bboxes, keypoints, scores, labels, label_names = self.compute(
                input_size=self.input_size,
                loc=loc,
                conf=conf,
                iou=iou,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                max_det=self.max_det,
                anchors=self._get_cached_anchors(),
                label_names=self.label_names,
                nms_fn=nms_cv2,
                top_left_wh_to_xywh_fn=top_left_wh_to_xywh,
            )
            self.emit(output, bboxes, keypoints, scores, labels, label_names)

    def extract(self, output: dai.NNData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            The localization, class-confidence, and IoU tensors, in that order. Missing
            configured names are inferred from unique ``loc``, ``conf``, and ``iou``
            prefixes.

        Raises:
            ValueError: If a configured layer is absent or inferred layer prefixes are
                missing or ambiguous.
        """
        output_layer_names = output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {output_layer_names}")

        if self.loc_output_layer_name:
            try:
                loc = output.getTensor(self.loc_output_layer_name, dequantize=True)
            except KeyError as err:
                raise ValueError(
                    f"Layer {self.loc_output_layer_name} not found in the model output."
                ) from err
        else:
            loc_output_layer_name_candidates = [
                layer_name
                for layer_name in output_layer_names
                if layer_name.startswith("loc")
            ]
            if len(loc_output_layer_name_candidates) == 0:
                raise ValueError("No loc layer candidates found in the model output.")
            if len(loc_output_layer_name_candidates) > 1:
                raise ValueError(
                    "Multiple loc layer candidates found in the model output."
                )
            self.loc_output_layer_name = loc_output_layer_name_candidates[0]
            loc = output.getTensor(self.loc_output_layer_name, dequantize=True)

        if self.conf_output_layer_name:
            try:
                conf = output.getTensor(self.conf_output_layer_name, dequantize=True)
            except KeyError as err:
                raise ValueError(
                    f"Layer {self.conf_output_layer_name} not found in the model output."
                ) from err
        else:
            conf_output_layer_name_candidates = [
                layer_name
                for layer_name in output_layer_names
                if layer_name.startswith("conf")
            ]
            if len(conf_output_layer_name_candidates) == 0:
                raise ValueError("No conf layer candidates found in the model output.")
            if len(conf_output_layer_name_candidates) > 1:
                raise ValueError(
                    "Multiple conf layer candidates found in the model output."
                )
            self.conf_output_layer_name = conf_output_layer_name_candidates[0]
            conf = output.getTensor(self.conf_output_layer_name, dequantize=True)

        if self.iou_output_layer_name:
            try:
                iou = output.getTensor(self.iou_output_layer_name, dequantize=True)
            except KeyError as err:
                raise ValueError(
                    f"Layer {self.iou_output_layer_name} not found in the model output."
                ) from err
        else:
            iou_output_layer_name_candidates = [
                layer_name
                for layer_name in output_layer_names
                if layer_name.startswith("iou")
            ]
            if len(iou_output_layer_name_candidates) == 0:
                raise ValueError("No iou layer candidates found in the model output.")
            if len(iou_output_layer_name_candidates) > 1:
                raise ValueError(
                    "Multiple iou layer candidates found in the model output."
                )
            self.iou_output_layer_name = iou_output_layer_name_candidates[0]
            iou = output.getTensor(self.iou_output_layer_name, dequantize=True)

        return loc, conf, iou

    @staticmethod
    def compute(
        *,
        input_size: tuple[int, int],
        loc: np.ndarray,
        conf: np.ndarray,
        iou: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        anchors: np.ndarray,
        label_names: list[str] | None = None,
        nms_fn: Callable[..., np.ndarray],
        top_left_wh_to_xywh_fn: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            input_size: Model input size as ``(width, height)``.
            loc: Per-anchor box and five-landmark offsets.
            conf: Per-anchor class-confidence tensor.
            iou: Per-anchor IoU confidence tensor.
            conf_threshold: Minimum detection confidence used to filter candidates.
            iou_threshold: Intersection-over-union threshold for non-maximum
                suppression.
            max_det: Maximum number of detection candidates to retain or consider during
                suppression.
            anchors: Precomputed anchor coordinates used to decode model predictions.
            label_names: Optional class-name lookup indexed by predicted class ID.
            nms_fn: Suppression callable accepting boxes, scores, confidence/IoU
                thresholds, and ``max_det``; returns retained indexes.
            top_left_wh_to_xywh_fn: Callable converting top-left XY/width/height boxes
                to center-XY/width/height.

        Returns:
            Normalized center-XY/width/height boxes, five normalized XY landmarks per
            face, scores, zero-valued class IDs, and optional class names. No candidates
            produces empty arrays.

        Note:
            Uses `depthai_nodes.node.parsers.utils.yunet.compute_yunet_detections`; see
            that helper for tensor layout and validation details.
        """
        return compute_yunet_detections(
            input_size=input_size,
            loc=loc,
            conf=conf,
            iou=iou,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            anchors=anchors,
            label_names=label_names,
            nms_fn=nms_fn,
            top_left_wh_to_xywh_fn=top_left_wh_to_xywh_fn,
        )

    def emit(
        self,
        output: dai.NNData,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        label_names: list[str] | None,
    ) -> None:
        """Create a ``dai.ImgDetections`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            bboxes: Normalized center-XY/width/height boxes returned by ``compute()``.
            keypoints: Normalized keypoint coordinates returned by ``compute()``.
            scores: Confidence scores corresponding to the computed payload.
            labels: Integer class IDs corresponding to the boxes.
            label_names: Optional class names corresponding to the detections.
        """
        detections_message = create_detection_message(
            bboxes=bboxes,
            scores=scores,
            keypoints=keypoints,
            labels=labels,
            label_names=label_names,
        )

        detections_message.setTimestamp(output.getTimestamp())
        detections_message.setSequenceNum(output.getSequenceNum())
        detections_message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            detections_message.setTransformation(transformation)

        self._logger.debug(f"Created detections message with {len(bboxes)} faces")
        self.out.send(detections_message)
        self._logger.debug("Detections message sent successfully")
