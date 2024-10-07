from typing import Any, Dict, Tuple

import depthai as dai

from ..messages.creators import create_detection_message
from .detection import DetectionParser
from .utils.nms import nms_cv2
from .utils.yunet import decode_detections, format_detections, prune_detections


class YuNetParser(DetectionParser):
    """Parser class for parsing the output of the YuNet face detection model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    conf_threshold : float
        Confidence score threshold for detected faces.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.

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
        input_shape: Tuple[int, int] = None,
        loc_output_layer_name: str = None,
        conf_output_layer_name: str = None,
        iou_output_layer_name: str = None,
    ) -> None:
        """Initializes the YuNetParser node.

        @param conf_threshold: Confidence score threshold for detected faces.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param input_shape: Input shape of the model (width, height).
        @type input_shape: Tuple[int, int]
        @param loc_output_layer_name: Output layer name for the location predictions.
        @type loc_output_layer_name: str
        @param conf_output_layer_name: Output layer name for the confidence predictions.
        @type conf_output_layer_name: str
        @param iou_output_layer_name: Output layer name for the IoU predictions.
        @type iou_output_layer_name: str
        """
        super().__init__("", conf_threshold, iou_threshold, max_det)

        self.loc_output_layer_name = loc_output_layer_name
        self.conf_output_layer_name = conf_output_layer_name
        self.iou_output_layer_name = iou_output_layer_name
        self.input_shape = input_shape

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "YuNetParser":
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        YuNetParser
            Returns the parser object with the head configuration set.
        """
        output_layers = head_config["outputs"]
        if len(output_layers) != 3:
            raise ValueError(
                f"YuNetParser expects exactly 3 output layers, got {output_layers} layers."
            )
        for output_layer in output_layers:
            self.loc_output_layer_name = output_layer if "loc" in output_layer else None
            self.conf_output_layer_name = (
                output_layer if "conf" in output_layer else None
            )
            self.iou_output_layer_name = output_layer if "iou" in output_layer else None

        self.conf_threshold = head_config["metadata"]["conf_threshold"]
        self.iou_threshold = head_config["metadata"]["iou_threshold"]
        self.max_det = head_config["metadata"]["max_det"]

        return self

    def setInputShape(self, width, height):
        """Sets the input shape.

        @param height: Height of the input image.
        @type height: int
        @param width: Width of the input image.
        @type width: int
        """
        self.input_shape = (width, height)

    def setOutputLayerNames(
        self, loc_output_layer_name, conf_output_layer_name, iou_output_layer_name
    ):
        """Sets the output layers.

        @param loc_output_layer_name: Output layer name for the location predictions.
        @type loc_output_layer_name: str
        @param conf_output_layer_name: Output layer name for the confidence predictions.
        @type conf_output_layer_name: str
        @param iou_output_layer_name: Output layer name for the IoU predictions.
        @type iou_output_layer_name: str
        """
        self.loc_output_layer_name = loc_output_layer_name
        self.conf_output_layer_name = conf_output_layer_name
        self.iou_output_layer_name = iou_output_layer_name

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()

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

            # decode detections
            bboxes, keypoints, scores = decode_detections(
                input_shape=self.input_shape,
                loc=loc,
                conf=conf,
                iou=iou,
            )

            # prune detections
            bboxes, keypoints, scores = prune_detections(
                bboxes=bboxes,
                keypoints=keypoints,
                scores=scores,
                conf_threshold=self.conf_threshold,
            )

            # format detections
            bboxes, keypoints, scores = format_detections(
                bboxes=bboxes,
                keypoints=keypoints,
                scores=scores,
                input_shape=self.input_shape,
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
            keypoints = keypoints[keep_indices]
            scores = scores[keep_indices]

            detections_message = create_detection_message(
                bboxes=bboxes, scores=scores, keypoints=keypoints
            )

            detections_message.setTimestamp(output.getTimestamp())

            self.out.send(detections_message)
