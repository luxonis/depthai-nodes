from typing import Any, Dict, List, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils import xyxy_to_xywh
from depthai_nodes.node.parsers.utils.scrfd import compute_anchor_centers, decode_scrfd


class SCRFDParser(DetectionParser):
    """Parser class for parsing the output of the SCRFD face detection model.

    Attributes
    ----------
    output_layer_name: List[str]
        Names of the output layers relevant to the parser.
    conf_threshold : float
        Confidence score threshold for detected faces.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.
    input_size : tuple
        Input size of the model.
    feat_stride_fpn : tuple
        Tuple of the feature strides.
    num_anchors : int
        Number of anchors.

    Output Message/s
    ----------------
    **Type**: dai.ImgDetections

    **Description**: ImgDetections message containing bounding boxes, labels, and confidence scores of detected faces.
    """

    def __init__(
        self,
        output_layer_names: List[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_det: int = 100,
        input_size: Tuple[int, int] = (640, 640),
        feat_stride_fpn: Tuple = (8, 16, 32),
        num_anchors: int = 2,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_names: Names of the output layers relevant to the parser.
        @type output_layer_names: List[str]
        @param conf_threshold: Confidence score threshold for detected faces.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param input_size: Input size of the model.
        @type input_size: tuple
        @param feat_stride_fpn: List of the feature strides.
        @type feat_stride_fpn: tuple
        @param num_anchors: Number of anchors.
        @type num_anchors: int
        """
        super().__init__(conf_threshold, iou_threshold, max_det)
        self.output_layer_names = (
            [] if output_layer_names is None else output_layer_names
        )

        self.feat_stride_fpn = feat_stride_fpn
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.label_names = ["Face"]
        self._cached_anchors = compute_anchor_centers(
            self.feat_stride_fpn, self.input_size, self.num_anchors
        )
        self._logger.debug(
            f"SCRFDParser initialized with output_layer_names={output_layer_names}, conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, max_det={max_det}, input_size={input_size}, feat_stride_fpn={feat_stride_fpn}, num_anchors={num_anchors}"
        )

    def setOutputLayerNames(self, output_layer_names: List[str]) -> None:
        """Sets the output layer name(s) for the parser.

        @param output_layer_names: The name of the output layer(s) to be used.
        @type output_layer_names: List[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(layer, str) for layer in output_layer_names):
            raise ValueError("Output layer names must be a list of strings.")
        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def setInputSize(self, input_size: Tuple[int, int]) -> None:
        """Sets the input size of the model.

        @param input_size: Input size of the model.
        @type input_size: list
        """
        if not isinstance(input_size, tuple):
            raise ValueError("Input size must be a tuple.")
        if not all(isinstance(size, int) for size in input_size):
            raise ValueError("Input size must be a tuple of integers.")
        self.input_size = input_size
        self._logger.debug(f"Input size set to {self.input_size}")

    def setFeatStrideFPN(self, feat_stride_fpn: List[int]) -> None:
        """Sets the feature stride of the FPN.

        @param feat_stride_fpn: Feature stride of the FPN.
        @type feat_stride_fpn: list
        """
        if not isinstance(feat_stride_fpn, list):
            raise ValueError("Feature stride must be a list.")
        if not all(isinstance(stride, int) for stride in feat_stride_fpn):
            raise ValueError("Feature stride must be a list of integers.")
        self.feat_stride_fpn = feat_stride_fpn
        self._logger.debug(f"Feature stride set to {self.feat_stride_fpn}")

    def setNumAnchors(self, num_anchors: int) -> None:
        """Sets the number of anchors.

        @param num_anchors: Number of anchors.
        @type num_anchors: int
        """
        if not isinstance(num_anchors, int):
            raise ValueError("Number of anchors must be an integer.")
        self.num_anchors = num_anchors
        self._logger.debug(f"Number of anchors set to {self.num_anchors}")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "SCRFDParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: SCRFDParser
        """

        super().build(head_config)

        output_layers = head_config.get("outputs", [])
        score_layer_names = [layer for layer in output_layers if "score" in layer]
        bbox_layer_names = [layer for layer in output_layers if "bbox" in layer]
        kps_layer_names = [layer for layer in output_layers if "kps" in layer]

        if len(score_layer_names) != len(bbox_layer_names) or len(
            score_layer_names
        ) != len(kps_layer_names):
            raise ValueError(
                f"Number of score, bbox, and kps layers should be equal, got {len(score_layer_names)}, {len(bbox_layer_names)}, and {len(kps_layer_names)} layers."
            )

        self.output_layer_names = output_layers
        self.feat_stride_fpn = head_config.get("feat_stride_fpn", self.feat_stride_fpn)
        self.num_anchors = head_config.get("num_anchors", self.num_anchors)
        self._cached_anchors = compute_anchor_centers(
            self.feat_stride_fpn, self.input_size, self.num_anchors
        )

        self._logger.debug(
            f"SCRFDParser built with output_layer_names={self.output_layer_names}, feat_stride_fpn={self.feat_stride_fpn}, num_anchors={self.num_anchors}"
        )

        return self

    def run(self):
        self._logger.debug("SCRFDParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            scores_concatenated = []
            bboxes_concatenated = []
            kps_concatenated = []

            if len(self.output_layer_names) == 0:
                self.output_layer_names = output.getAllLayerNames()

            self._logger.debug(
                f"Processing input with layers: {output.getAllLayerNames()}"
            )

            for stride in self.feat_stride_fpn:
                score_layer_name = f"score_{stride}"
                bbox_layer_name = f"bbox_{stride}"
                kps_layer_name = f"kps_{stride}"
                if score_layer_name not in self.output_layer_names:
                    raise ValueError(
                        f"Layer {score_layer_name} not found in the model output."
                    )
                if bbox_layer_name not in self.output_layer_names:
                    raise ValueError(
                        f"Layer {bbox_layer_name} not found in the model output."
                    )
                if kps_layer_name not in self.output_layer_names:
                    raise ValueError(
                        f"Layer {kps_layer_name} not found in the model output."
                    )

                score_tensor = (
                    output.getTensor(score_layer_name, dequantize=True)
                    .flatten()
                    .astype(np.float32)
                )
                bbox_tensor = (
                    output.getTensor(bbox_layer_name, dequantize=True)
                    .reshape(len(score_tensor), 4)
                    .astype(np.float32)
                )
                kps_tensor = (
                    output.getTensor(kps_layer_name, dequantize=True)
                    .reshape(len(score_tensor), 10)
                    .astype(np.float32)
                )

                scores_concatenated.append(score_tensor)
                bboxes_concatenated.append(bbox_tensor)
                kps_concatenated.append(kps_tensor)

            bboxes, scores, keypoints = decode_scrfd(
                bboxes_concatenated=bboxes_concatenated,
                scores_concatenated=scores_concatenated,
                kps_concatenated=kps_concatenated,
                feat_stride_fpn=self.feat_stride_fpn,
                input_size=self.input_size,
                num_anchors=self.num_anchors,
                score_threshold=self.conf_threshold,
                nms_threshold=self.iou_threshold,
                anchors=self._cached_anchors,
            )
            bboxes = xyxy_to_xywh(bboxes)
            bboxes = np.clip(bboxes, 0, 1)

            labels = np.array([0] * len(bboxes))

            label_names = (
                [self.label_names[label] for label in labels]
                if self.label_names
                else None
            )
            message = create_detection_message(
                bboxes=bboxes,
                scores=scores,
                labels=labels,
                label_names=label_names,
                keypoints=keypoints,
            )
            message.setTimestamp(output.getTimestamp())
            message.setSequenceNum(output.getSequenceNum())
            message.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                message.setTransformation(transformation)

            self._logger.debug(
                f"Created detection message with {len(bboxes)} detections"
            )

            self.out.send(message)

            self._logger.debug("Detection message sent successfully")
