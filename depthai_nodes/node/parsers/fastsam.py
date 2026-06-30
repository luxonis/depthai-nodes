from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_segmentation_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.fastsam import (
    box_prompt,
    build_mask_coeffs,
    compute_fastsam_mask,
    decode_fastsam_output,
    merge_masks,
    point_prompt,
    process_masks,
)
from depthai_nodes.node.parsers.utils.masks_utils import get_segmentation_outputs


class FastSAMParser(BaseParser):
    """Parser class for parsing the output of the FastSAM model.

    Attributes
    ----------
    conf_threshold : float
        Confidence score threshold for detected faces.
    n_classes : int
        Number of classes in the model.
    iou_threshold : float
        Non-maximum suppression threshold.
    mask_conf : float
        Mask confidence threshold.
    prompt : str
        Prompt type.
    points : tuple[int, int]
        Points.
    point_label : int
        Point label.
    bbox : tuple[int, int, int, int]
        Bounding box.
    yolo_outputs : list[str]
        Names of the YOLO outputs.
    mask_outputs : list[str]
        Names of the mask outputs.
    protos_output : str
        Name of the protos output.

    Output Message/s
    ----------------
    **Type**: SegmentationMask

    **Description**: SegmentationMask message containing the resulting segmentation masks given the prompt.

    Error Handling
    --------------
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        n_classes: int = 1,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        prompt: str = "everything",
        points: tuple[int, int] | None = None,
        point_label: int | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        yolo_outputs: list[str] = None,
        mask_outputs: list[str] = None,
        protos_output: str = "protos_output",
    ) -> None:
        """Initializes the parser node.

        @param conf_threshold: The confidence threshold for the detections
        @type conf_threshold: float
        @param n_classes: The number of classes in the model
        @type n_classes: int
        @param iou_threshold: The intersection over union threshold
        @type iou_threshold: float
        @param mask_conf: The mask confidence threshold
        @type mask_conf: float
        @param prompt: The prompt type
        @type prompt: str
        @param points: The points
        @type points: tuple[int, int] | None
        @param point_label: The point label
        @type point_label: int | None
        @param bbox: The bounding box
        @type bbox: tuple[int, int, int, int] | None
        @param yolo_outputs: The YOLO outputs
        @type yolo_outputs: list[str]
        @param mask_outputs: The mask outputs
        @type mask_outputs: list[str]
        @param protos_output: The protos output
        @type protos_output: str
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.mask_conf = mask_conf
        self.prompt = prompt
        self.points = points
        self.point_label = point_label
        self.bbox = bbox
        self.yolo_outputs = (
            ["output1_yolov8", "output2_yolov8", "output3_yolov8"]
            if yolo_outputs is None
            else yolo_outputs
        )
        self.mask_outputs = (
            ["output1_masks", "output2_masks", "output3_masks"]
            if mask_outputs is None
            else mask_outputs
        )
        self.protos_output = protos_output
        self._logger.debug(
            f"FastSAMParser initialized with conf_threshold={conf_threshold}, n_classes={n_classes}, iou_threshold={iou_threshold}, mask_conf={mask_conf}, prompt='{prompt}', points={points}, point_label={point_label}, bbox={bbox}, yolo_outputs={yolo_outputs}, mask_outputs={mask_outputs}, protos_output='{protos_output}'"
        )

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold.

        @param threshold: Confidence score threshold.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.conf_threshold = threshold
        self._logger.debug(f"Confidence threshold set to {self.conf_threshold}")

    def setNumClasses(self, n_classes: int) -> None:
        """Sets the number of classes in the model.

        @param numClasses: The number of classes in the model.
        @type numClasses: int
        """
        if not isinstance(n_classes, int):
            raise ValueError("Number of classes must be an integer.")
        self.n_classes = n_classes
        self._logger.debug(f"Number of classes set to {self.n_classes}")

    def setIouThreshold(self, iou_threshold: float) -> None:
        """Sets the intersection over union threshold.

        @param iou_threshold: The intersection over union threshold.
        @type iou_threshold: float
        """
        if not isinstance(iou_threshold, float):
            raise ValueError("IOU threshold must be a float.")
        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError("IOU threshold must be between 0 and 1.")
        self.iou_threshold = iou_threshold
        self._logger.debug(
            f"Intersection over union threshold set to {self.iou_threshold}"
        )

    def setMaskConfidence(self, mask_conf: float) -> None:
        """Sets the mask confidence threshold.

        @param mask_conf: The mask confidence threshold.
        @type mask_conf: float
        """
        if not isinstance(mask_conf, float):
            raise ValueError("Mask confidence must be a float.")
        if mask_conf < 0 or mask_conf > 1:
            raise ValueError("Mask confidence must be between 0 and 1.")
        self.mask_conf = mask_conf
        self._logger.debug(f"Mask confidence threshold set to {self.mask_conf}")

    def setPrompt(self, prompt: str) -> None:
        """Sets the prompt type.

        @param prompt: The prompt type
        @type prompt: str
        """
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        if prompt not in ["everything", "bbox", "point"]:
            raise ValueError("Prompt must be one of 'everything', 'bbox', or 'point'")
        self.prompt = prompt
        self._logger.debug(f"Prompt set to '{self.prompt}'")

    def setPoints(self, points: tuple[int, int]) -> None:
        """Sets the points.

        @param points: The points
        @type points: tuple[int, int]
        """
        if not isinstance(points, tuple):
            raise ValueError("Points must be a tuple.")
        if len(points) != 2:
            raise ValueError("Points must be a tuple of length 2.")
        if not all(isinstance(p, int) for p in points):
            raise ValueError("Point elements must be integers.")
        self.points = points
        self._logger.debug(f"Points set to {self.points}")

    def setPointLabel(self, point_label: int) -> None:
        """Sets the point label.

        @param point_label: The point label
        @type point_label: int
        """
        if not isinstance(point_label, int):
            raise ValueError("Point label must be an integer.")
        self.point_label = point_label
        self._logger.debug(f"Point label set to {self.point_label}")

    def setBoundingBox(self, bbox: tuple[int, int, int, int]) -> None:
        """Sets the bounding box.

        @param bbox: The bounding box
        @type bbox: tuple[int, int, int, int]
        """
        if not isinstance(bbox, tuple):
            raise ValueError("Bounding box must be a tuple.")
        if len(bbox) != 4:
            raise ValueError("Bounding box must be a tuple of length 4.")
        if not all(isinstance(b, int) for b in bbox):
            raise ValueError("Bounding box elements must be integers.")
        self.bbox = bbox
        self._logger.debug(f"Bounding box set to {self.bbox}")

    def setYoloOutputs(self, yolo_outputs: list[str]) -> None:
        """Sets the YOLO outputs.

        @param yolo_outputs: The YOLO outputs
        @type yolo_outputs: list[str]
        """
        if not isinstance(yolo_outputs, list):
            raise ValueError("YOLO outputs must be a list.")
        if not all(isinstance(o, str) for o in yolo_outputs):
            raise ValueError("YOLO outputs must be a list of strings.")
        self.yolo_outputs = yolo_outputs
        self._logger.debug(f"YOLO outputs set to {self.yolo_outputs}")

    def setMaskOutputs(self, mask_outputs: list[str]) -> None:
        """Sets the mask outputs.

        @param mask_outputs: The mask outputs
        @type mask_outputs: list[str]
        """
        if not isinstance(mask_outputs, list):
            raise ValueError("Mask outputs must be a list.")
        if not all(isinstance(o, str) for o in mask_outputs):
            raise ValueError("Mask outputs must be a list of strings.")
        self.mask_outputs = mask_outputs
        self._logger.debug(f"Mask outputs set to {self.mask_outputs}")

    def setProtosOutput(self, protos_output: str) -> None:
        """Sets the protos output.

        @param protos_output: The protos output
        @type protos_output: str
        """
        if not isinstance(protos_output, str):
            raise ValueError("Protos output must be a string.")
        self.protos_output = protos_output
        self._logger.debug(f"Protos output set to '{self.protos_output}'")

    def build(self, head_config: dict[str, Any]) -> "FastSAMParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: FastSAMParser
        """

        output_layers = head_config["outputs"]
        yolo_outputs = [name for name in output_layers if "_yolo" in name]
        if len(yolo_outputs) != 0:
            self.yolo_outputs = yolo_outputs

        mask_outputs = [name for name in output_layers if "_masks" in name]
        if len(mask_outputs) != 0:
            self.mask_outputs = mask_outputs

        self.protos_output = head_config.get("protos_output", self.protos_output)

        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.n_classes = head_config.get("n_classes", self.n_classes)
        self.iou_threshold = head_config.get("iou_threshold", self.iou_threshold)
        self.mask_conf = head_config.get("mask_conf", self.mask_conf)
        self.prompt = head_config.get("prompt", self.prompt)
        self.points = head_config.get("points", self.points)
        self.point_label = head_config.get("point_label", self.point_label)
        self.bbox = head_config.get("bbox", self.bbox)

        self._logger.debug(
            f"FastSAMParser built with conf_threshold={self.conf_threshold}, n_classes={self.n_classes}, iou_threshold={self.iou_threshold}, mask_conf={self.mask_conf}, prompt='{self.prompt}', points={self.points}, point_label={self.point_label}, bbox={self.bbox}, yolo_outputs={self.yolo_outputs}, mask_outputs={self.mask_outputs}, protos_output='{self.protos_output}'"
        )

        return self

    def run(self):
        self._logger.debug("FastSAMParser run started")
        if self.prompt not in ["everything", "bbox", "point"]:
            raise ValueError("Prompt must be one of 'everything', 'bbox', or 'point'")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data

            (
                outputs_values,
                masks_outputs_values,
                protos_output,
                protos_len,
            ) = self.extract(output)
            results_masks = self.compute(
                outputs_values,
                masks_outputs_values,
                protos_output,
                protos_len,
                conf_threshold=self.conf_threshold,
                n_classes=self.n_classes,
                iou_threshold=self.iou_threshold,
                mask_conf=self.mask_conf,
                prompt=self.prompt,
                points=self.points,
                point_label=self.point_label,
                bbox=self.bbox,
            )
            self.emit(output, results_masks)

    def extract(
        self, output: dai.NNData
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, int]:
        outputs_names = sorted([name for name in self.yolo_outputs])
        self._logger.debug(f"Processing input with layers: {outputs_names}")
        outputs_values = [
            output.getTensor(
                o, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
            ).astype(np.float32)
            for o in outputs_names
        ]

        (
            masks_outputs_values,
            protos_output,
            protos_len,
        ) = get_segmentation_outputs(output, self.mask_outputs, self.protos_output)
        return outputs_values, masks_outputs_values, protos_output, protos_len

    @staticmethod
    def compute(
        outputs_values: list[np.ndarray],
        masks_outputs_values: list[np.ndarray],
        protos_output: np.ndarray,
        protos_len: int,
        *,
        conf_threshold: float,
        n_classes: int,
        iou_threshold: float,
        mask_conf: float,
        prompt: str,
        points: tuple[int, int] | None,
        point_label: int | None,
        bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        return compute_fastsam_mask(
            outputs_values,
            masks_outputs_values,
            protos_output,
            protos_len,
            conf_threshold=conf_threshold,
            n_classes=n_classes,
            iou_threshold=iou_threshold,
            mask_conf=mask_conf,
            prompt=prompt,
            points=points,
            point_label=point_label,
            bbox=bbox,
        )

    def emit(self, output: dai.NNData, results_masks: np.ndarray) -> None:
        segmentation_message = create_segmentation_message(results_masks)
        transformation = output.getTransformation()
        if transformation is not None:
            segmentation_message.setTransformation(transformation)
        segmentation_message.setTimestamp(output.getTimestamp())
        segmentation_message.setSequenceNum(output.getSequenceNum())
        segmentation_message.setTimestampDevice(output.getTimestampDevice())

        self._logger.debug(
            f"Created segmentation message with {len(results_masks)} masks"
        )
        self.out.send(segmentation_message)
        self._logger.debug("Segmentation message sent successfully")
