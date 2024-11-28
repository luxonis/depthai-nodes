from typing import Any, Dict, List, Optional

import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .base_parser import BaseParser
from .utils.bbox_format_converters import normalize_bboxes, xyxy_to_xywh
from .utils.masks_utils import (
    get_segmentation_outputs,
    process_single_mask,
)
from .utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
    parse_kpts,
)


class YOLOExtendedParser(BaseParser):
    """Parser class for parsing the output of the YOLO Instance Segmentation and Pose
    Estimation models.

    Attributes
    ----------
    conf_threshold : float
        Confidence score threshold for detected faces.
    n_classes : int
        Number of classes in the model.
    iou_threshold : float
        Intersection over union threshold.
    mask_conf : float
        Mask confidence threshold.
    n_keypoints : int
        Number of keypoints in the model.
    anchors : Optional[List[List[List[float]]]]
        Anchors for the YOLO model (optional).
    subtype : str
        Version of the YOLO model.


    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints or masks and protos of the detected objects.
    """

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(
        self,
        conf_threshold: float = 0.5,
        n_classes: int = 1,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        n_keypoints: int = 17,
        anchors: Optional[List[List[List[float]]]] = None,
        subtype: str = "",
    ):
        """Initialize the parser node.

        @param conf_threshold: The confidence threshold for the detections
        @type conf_threshold: float
        @param n_classes: The number of classes in the model
        @type n_classes: int
        @param iou_threshold: The intersection over union threshold
        @type iou_threshold: float
        @param mask_conf: The mask confidence threshold
        @type mask_conf: float
        @param n_keypoints: The number of keypoints in the model
        @type n_keypoints: int
        @param anchors: The anchors for the YOLO model
        @type anchors: Optional[List[List[List[float]]]]
        @param subtype: The version of the YOLO model
        @type subtype: str
        """
        super().__init__()

        self.output_layer_names = []
        self.conf_threshold = conf_threshold
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.mask_conf = mask_conf
        self.n_keypoints = n_keypoints
        self.anchors = anchors
        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO version {subtype}. Supported YOLO versions are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

    def setOutputLayerNames(self, output_layer_names: List[str]) -> None:
        """Sets the output layer names for the parser.

        @param output_layer_names: The output layer names for the parser.
        @type output_layer_names: List[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(layer, str) for layer in output_layer_names):
            raise ValueError("Output layer names must be a list of strings.")

        self.output_layer_names = output_layer_names

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence score threshold must be a float.")

        self.conf_threshold = threshold

    def setNumClasses(self, n_classes: int) -> None:
        """Sets the number of classes in the model.

        @param numClasses: The number of classes in the model.
        @type numClasses: int
        """
        if not isinstance(n_classes, int):
            raise ValueError("Number of classes must be an integer.")
        if n_classes < 1:
            raise ValueError("Number of classes must be greater than 0.")

        self.n_classes = n_classes

    def setIouThreshold(self, iou_threshold: float) -> None:
        """Sets the intersection over union threshold.

        @param iou_threshold: The intersection over union threshold.
        @type iou_threshold: float
        """
        if not isinstance(iou_threshold, float):
            raise ValueError("Intersection over union threshold must be a float.")
        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError(
                "Intersection over union threshold must be between 0 and 1."
            )

        self.iou_threshold = iou_threshold

    def setMaskConfidence(self, mask_conf: float) -> None:
        """Sets the mask confidence threshold.

        @param mask_conf: The mask confidence threshold.
        @type mask_conf: float
        """
        if not isinstance(mask_conf, float):
            raise ValueError("Mask confidence threshold must be a float.")
        if mask_conf < 0 or mask_conf > 1:
            raise ValueError("Mask confidence threshold must be between 0 and 1.")

        self.mask_conf = mask_conf

    def setNumKeypoints(self, n_keypoints: int) -> None:
        """Sets the number of keypoints in the model.

        @param n_keypoints: The number of keypoints in the model.
        @type n_keypoints: int
        """
        if not isinstance(n_keypoints, int):
            raise ValueError("Number of keypoints must be an integer.")
        if n_keypoints < 1:
            raise ValueError("Number of keypoints must be greater than 0.")

        self.n_keypoints = n_keypoints

    def setAnchors(self, anchors: List[List[List[float]]]) -> None:
        """Sets the anchors for the YOLO model.

        @param anchors: The anchors for the YOLO model.
        @type anchors: List[List[List[float]]]
        """
        for anchor in anchors:
            if not isinstance(anchor, list):
                raise ValueError("Anchors must be a list of lists.")
            if not all(isinstance(val, list) for val in anchor):
                raise ValueError("Anchors must be a list of lists of lists.")
            if not all(isinstance(val, float) for sublist in anchor for val in sublist):
                raise ValueError("Anchors must be a list of lists of floats.")

        self.anchors = anchors

    def setSubtype(self, subtype: str) -> None:
        """Sets the subtype of the YOLO model.

        @param subtype: The subtype of the YOLO model.
        @type subtype: YOLOSubtype
        """
        if not isinstance(subtype, str):
            raise ValueError("Subtype must be a string.")

        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

    def build(
        self,
        head_config: Dict[str, Any],
    ):
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: YOLOExtendedParser
        """

        output_layers = head_config.get("outputs", [])
        bbox_layer_names = [name for name in output_layers if "_yolo" in name]
        kps_layer_names = [name for name in output_layers if "kpt_output" in name]
        masks_layer_names = [name for name in output_layers if "_masks" in name]

        if not bbox_layer_names:
            raise ValueError(
                "YOLOExtendedParser requires the output layers of the bounding boxes detection head."
            )
        if (kps_layer_names and masks_layer_names) or (
            not bbox_layer_names and (kps_layer_names or masks_layer_names)
        ):
            raise ValueError(
                "YOLOExtendedParser requires either the output layers of the keypoints detection head or the output layers of the masks detection head along with the output layers of the bounding boxes detection head but not both."
            )

        self.output_layer_names = bbox_layer_names + kps_layer_names + masks_layer_names

        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.n_classes = head_config.get("n_classes", self.n_classes)
        self.iou_threshold = head_config.get("iou_threshold", self.iou_threshold)
        self.mask_conf = head_config.get("mask_conf", self.mask_conf)
        self.anchors = head_config.get("anchors", self.anchors)
        self.n_keypoints = head_config.get("n_keypoints", self.n_keypoints)
        subtype = head_config.get("subtype", self.subtype)
        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

        return self

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data
            # Get all the layer names
            layer_names = self.output_layer_names or output.getAllLayerNames()

            outputs_names = sorted(
                [name for name in layer_names if "_yolo" in name or "yolo-" in name]
            )
            outputs_values = [
                output.getTensor(
                    o, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
                ).astype(np.float32)
                for o in outputs_names
            ]

            if (
                any("kpt_output" in name for name in layer_names)
                and self.subtype != YOLOSubtype.P
            ):
                mode = self._KPTS_MODE
                # Get the keypoint outputs
                kpts_output_names = sorted(
                    [name for name in layer_names if "kpt_output" in name]
                )
                kpts_outputs = [
                    output.getTensor(o, dequantize=True).astype(np.float32)
                    for o in kpts_output_names
                ]
            elif (
                any("_masks" in name for name in layer_names)
                and self.subtype != YOLOSubtype.P
            ):
                mode = self._SEG_MODE
                # Get the segmentation outputs
                (
                    masks_outputs_values,
                    protos_output,
                    protos_len,
                ) = get_segmentation_outputs(output)
            else:
                mode = self._DET_MODE

            # Get the model's input shape
            strides = (
                [8, 16, 32]
                if self.subtype
                not in [YOLOSubtype.V3UT, YOLOSubtype.V3T, YOLOSubtype.V4T]
                else [16, 32]
            )
            input_shape = tuple(
                dim * strides[0] for dim in outputs_values[0].shape[2:4]
            )

            # Reshape the anchors based on the model's output heads
            if self.anchors is not None:
                self.anchors = np.array(self.anchors).reshape(len(strides), -1)

            # Decode the outputs
            results = decode_yolo_output(
                outputs_values,
                strides,
                self.anchors,
                kpts=kpts_outputs if mode == self._KPTS_MODE else None,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                num_classes=self.n_classes,
                det_mode=mode == self._DET_MODE,
                subtype=self.subtype,
            )

            bboxes, labels, scores, additional_output = [], [], [], []
            final_mask = np.full(input_shape, -1, dtype=np.int16)
            for i in range(results.shape[0]):
                bbox, conf, label, other = (
                    results[i, :4],
                    results[i, 4],
                    results[i, 5].astype(int),
                    results[i, 6:],
                )

                bbox = xyxy_to_xywh(bbox.reshape(1, 4))
                bbox = normalize_bboxes(
                    bbox, height=input_shape[0], width=input_shape[1]
                )[0]
                bboxes.append(bbox)
                labels.append(int(label))
                scores.append(conf)

                if mode == self._KPTS_MODE:
                    kpts = parse_kpts(other, self.n_keypoints, input_shape)
                    additional_output.append(kpts)
                elif mode == self._SEG_MODE:
                    seg_coeff = other.astype(int)
                    hi, ai, xi, yi = seg_coeff
                    mask_coeff = masks_outputs_values[hi][
                        0, ai * protos_len : (ai + 1) * protos_len, yi, xi
                    ]
                    mask = process_single_mask(
                        protos_output[0], mask_coeff, self.mask_conf, bbox
                    )
                    # Resize mask to input shape
                    resized_mask = cv2.resize(
                        mask,
                        (input_shape[1], input_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # Fill the final mask with the instance values
                    final_mask[resized_mask > 0] = i

            bboxes = np.array(bboxes)
            bboxes = np.clip(bboxes, 0, 1)

            if mode == self._KPTS_MODE:
                additional_output = np.array(additional_output)
                keypoints = np.array([])
                keypoints_scores = np.array([])
                if additional_output.size > 0:
                    keypoints = additional_output[:, :, :2]
                    keypoints_scores = additional_output[:, :, 2]

                keypoints = np.clip(keypoints, 0, 1)
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                    keypoints=keypoints,
                    keypoints_scores=keypoints_scores,
                )
            elif mode == self._SEG_MODE:
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                    masks=final_mask,
                )
            else:
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                )

            detections_message.setTimestamp(output.getTimestamp())
            detections_message.transformation = output.getTransformation()

            self.out.send(detections_message)
