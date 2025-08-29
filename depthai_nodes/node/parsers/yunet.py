from typing import Any, Dict, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils import top_left_wh_to_xywh
from depthai_nodes.node.parsers.utils.nms import nms_cv2
from depthai_nodes.node.parsers.utils.yunet import (
    decode_and_prune_detections,
    format_detections,
    generate_anchors,
)


class YuNetParser(DetectionParser):
    """Parser class for parsing the output of the YuNet face detection model.

    Attributes
    ----------
    conf_threshold : float
        Confidence score threshold for detected faces.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.
    input_size : Tuple[int, int]
        Input size (width, height).
    loc_output_layer_name: str
        Name of the output layer containing the location predictions.
    conf_output_layer_name: str
        Name of the output layer containing the confidence predictions.
    iou_output_layer_name: str
        Name of the output layer containing the IoU predictions.

    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints of detected faces.
    """

    def __init__(
        self,
        conf_threshold: float = 0.8,
        iou_threshold: float = 0.3,
        max_det: int = 5000,
        input_size: Tuple[int, int] = None,
        loc_output_layer_name: str = None,
        conf_output_layer_name: str = None,
        iou_output_layer_name: str = None,
    ) -> None:
        """Initializes the parser node.

        @param conf_threshold: Confidence score threshold for detected faces.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param input_size: Input size of the model (width, height).
        @type input_size: Tuple[int, int]
        @param loc_output_layer_name: Output layer name for the location predictions.
        @type loc_output_layer_name: str
        @param conf_output_layer_name: Output layer name for the confidence predictions.
        @type conf_output_layer_name: str
        @param iou_output_layer_name: Output layer name for the IoU predictions.
        @type iou_output_layer_name: str
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

    def setInputSize(self, input_size: Tuple[int, int]) -> None:
        """Sets the input size of the model.

        @param input_size: Input size of the model (width, height).
        @type input_size: list
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

        @param loc_output_layer_name: Output layer name for the loc tensor.
        @type loc_output_layer_name: str
        """
        if not isinstance(loc_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.loc_output_layer_name = loc_output_layer_name
        self._logger.debug(
            f"Location output layer name set to '{self.loc_output_layer_name}'"
        )

    def setOutputLayerConf(self, conf_output_layer_name: str) -> None:
        """Sets the name of the output layer containing the confidence predictions.

        @param conf_output_layer_name: Output layer name for the conf tensor.
        @type conf_output_layer_name: str
        """
        if not isinstance(conf_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.conf_output_layer_name = conf_output_layer_name
        self._logger.debug(
            f"Confidence output layer name set to '{self.conf_output_layer_name}'"
        )

    def setOutputLayerIou(self, iou_output_layer_name: str) -> None:
        """Sets the name of the output layer containing the IoU predictions.

        @param iou_output_layer_name: Output layer name for the IoU tensor.
        @type iou_output_layer_name: str
        """
        if not isinstance(iou_output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.iou_output_layer_name = iou_output_layer_name
        self._logger.debug(
            f"IoU output layer name set to '{self.iou_output_layer_name}'"
        )

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "YuNetParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: YuNetParser
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
        self._logger.debug("YuNetParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {output_layer_names}")

            # get loc
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
                    if layer_name.startswith(("loc"))
                ]
                if len(loc_output_layer_name_candidates) == 0:
                    raise ValueError(
                        "No loc layer candidates found in the model output."
                    )
                elif len(loc_output_layer_name_candidates) > 1:
                    raise ValueError(
                        "Multiple loc layer candidates found in the model output."
                    )
                else:
                    self.loc_output_layer_name = loc_output_layer_name_candidates[0]

            # get conf
            if self.conf_output_layer_name:
                try:
                    conf = output.getTensor(
                        self.conf_output_layer_name, dequantize=True
                    )
                except KeyError as err:
                    raise ValueError(
                        f"Layer {self.conf_output_layer_name} not found in the model output."
                    ) from err
            else:
                conf_output_layer_name_candidates = [
                    layer_name
                    for layer_name in output_layer_names
                    if layer_name.startswith(("conf"))
                ]
                if len(conf_output_layer_name_candidates) == 0:
                    raise ValueError(
                        "No conf layer candidates found in the model output."
                    )
                elif len(conf_output_layer_name_candidates) > 1:
                    raise ValueError(
                        "Multiple conf layer candidates found in the model output."
                    )
                else:
                    self.conf_output_layer_name = conf_output_layer_name_candidates[0]

            # get iou
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
                    if layer_name.startswith(("iou"))
                ]
                if len(iou_output_layer_name_candidates) == 0:
                    raise ValueError(
                        "No iou layer candidates found in the model output."
                    )
                elif len(iou_output_layer_name_candidates) > 1:
                    raise ValueError(
                        "Multiple iou layer candidates found in the model output."
                    )
                else:
                    self.iou_output_layer_name = iou_output_layer_name_candidates[0]

            loc = output.getTensor(self.loc_output_layer_name, dequantize=True)
            conf = output.getTensor(self.conf_output_layer_name, dequantize=True)
            iou = output.getTensor(self.iou_output_layer_name, dequantize=True)

            # Use optimized combined decode and prune function
            anchors = self._get_cached_anchors()
            bboxes, keypoints, scores = decode_and_prune_detections(
                input_size=self.input_size,
                loc=loc,
                conf=conf,
                iou=iou,
                conf_threshold=self.conf_threshold,
                anchors=anchors,
            )

            # Skip further processing if no detections
            if len(bboxes) == 0:
                detections_message = create_detection_message(
                    bboxes=np.array([]),
                    scores=np.array([]),
                    keypoints=np.array([]),
                    labels=np.array([]),
                    label_names=[],
                )
                detections_message.setTimestamp(output.getTimestamp())
                detections_message.setTimestampDevice(output.getTimestampDevice())
                transformation = output.getTransformation()
                if transformation is not None:
                    detections_message.setTransformation(transformation)
                detections_message.setSequenceNum(output.getSequenceNum())
                self.out.send(detections_message)
                continue

            # format detections
            bboxes, keypoints, scores = format_detections(
                bboxes=bboxes,
                keypoints=keypoints,
                scores=scores,
                input_size=self.input_size,
            )

            # run nms
            keep_indices = nms_cv2(
                bboxes=bboxes,
                scores=scores,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                max_det=self.max_det,
            )

            bboxes = bboxes[keep_indices]
            bboxes = top_left_wh_to_xywh(bboxes)
            keypoints = keypoints[keep_indices]
            scores = scores[keep_indices]

            bboxes = np.clip(bboxes, 0, 1)
            keypoints = np.clip(keypoints, 0, 1)

            labels = np.array([0] * len(bboxes))

            label_names = (
                [self.label_names[label] for label in labels]
                if self.label_names
                else None
            )

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
