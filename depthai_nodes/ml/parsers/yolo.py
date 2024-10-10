from typing import Any, Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .base_parser import BaseParser
from .utils.bbox_format_converters import xyxy_to_xywh_norm
from .utils.yolo import YOLOVersion, decode_yolo_output, parse_kpts, process_single_mask


class YOLOExtendedParser(BaseParser):
    """Parser class for parsing the output of the YOLO Instance Segmentation and Pose
    Estimation models.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
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
    anchors : Optional[List[np.ndarray]]
        Anchors for the YOLO model (optional).
    yolo_version : str
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
        anchors: Optional[List[np.ndarray]] = None,
        yolo_version: str = "",
    ):
        """Initialize the YOLOExtendedParser node.

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
        @type anchors: Optional[List[np.ndarray]]
        @param yolo_version: The version of the YOLO model
        @type yolo_version: str
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
            self.yolo_version = YOLOVersion(yolo_version.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO version {yolo_version}. Supported YOLO versions are {[e.value for e in YOLOVersion][:-1]}."
            ) from err

    def build(
        self,
        head_config: Dict[str, Any],
    ):
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.
        Returns
        -------
        YOLOExtendedParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
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

        metadata = head_config["metadata"]
        self.conf_threshold = metadata["conf_threshold"]
        self.n_classes = metadata["n_classes"]
        self.iou_threshold = metadata["iou_threshold"]
        self.anchors = metadata["anchors"]
        if "mask_conf" in metadata:
            self.mask_conf = metadata["mask_conf"]
        if "n_keypoints" in metadata:
            self.n_keypoints = metadata["n_keypoints"]
        if "yolo_version" in metadata:
            try:
                self.yolo_version = YOLOVersion(metadata["yolo_version"].lower())
            except ValueError as err:
                raise ValueError(
                    f"Invalid YOLO version {metadata['yolo_version']}. Supported YOLO versions are {[e.value for e in YOLOVersion][:-1]}."
                ) from err
        return self

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        self.conf_threshold = threshold

    def setNumClasses(self, n_classes: int) -> None:
        """Sets the number of classes in the model.

        @param numClasses: The number of classes in the model.
        @type numClasses: int
        """
        self.n_classes = n_classes

    def setIouThreshold(self, iou_threshold: float) -> None:
        """Sets the intersection over union threshold.

        @param iou_threshold: The intersection over union threshold.
        @type iou_threshold: float
        """
        self.iou_threshold = iou_threshold

    def setMaskConfidence(self, mask_conf: float) -> None:
        """Sets the mask confidence threshold.

        @param mask_conf: The mask confidence threshold.
        @type mask_conf: float
        """
        self.mask_conf = mask_conf

    def setNumKeypoints(self, n_keypoints: int) -> None:
        """Sets the number of keypoints in the model.

        @param n_keypoints: The number of keypoints in the model.
        @type n_keypoints: int
        """
        self.n_keypoints = n_keypoints

    def setAnchors(self, anchors: List[np.ndarray]) -> None:
        """Sets the anchors for the YOLO model.

        @param anchors: The anchors for the YOLO model.
        @type anchors: List[np.ndarray]
        """
        self.anchors = anchors

    def setYoloVersion(self, yolo_version: str) -> None:
        """Sets the version of the YOLO model.

        @param yolo_version: The version of the YOLO model.
        @type yolo_version: YOLOVersion
        """
        try:
            self.yolo_version = YOLOVersion(yolo_version.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO version {yolo_version}. Supported YOLO versions are {[e.value for e in YOLOVersion][:-1]}."
            ) from err

    def setOutputLayerNames(self, output_layer_names: List[str]) -> None:
        """Sets the output layer names for the parser.

        @param output_layer_names: The output layer names for the parser.
        @type output_layer_names: List[str]
        """
        self.output_layer_names = output_layer_names

    def _get_segmentation_outputs(
        self, output: dai.NNData
    ) -> Tuple[List[np.ndarray], np.ndarray, int]:
        """Get the segmentation outputs from the Neural Network data."""
        # Get all the layer names
        layer_names = self.output_layer_names or output.getAllLayerNames()
        mask_outputs = sorted([name for name in layer_names if "_masks" in name])
        masks_outputs_values = [
            output.getTensor(o, dequantize=True).astype(np.float32)
            for o in mask_outputs
        ]
        protos_output = output.getTensor("protos_output", dequantize=True).astype(
            np.float32
        )
        protos_len = protos_output.shape[1]
        return masks_outputs_values, protos_output, protos_len

    def _reshape_seg_outputs(
        self,
        protos_output: np.ndarray,
        protos_len: int,
        masks_outputs_values: List[np.ndarray],
    ) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        """Reshape the segmentation outputs."""
        protos_output = protos_output.transpose((0, 3, 1, 2))
        protos_len = protos_output.shape[1]
        masks_outputs_values = [o.transpose((0, 3, 1, 2)) for o in masks_outputs_values]
        return protos_output, protos_len, masks_outputs_values

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
                output.getTensor(o, dequantize=True).astype(np.float32)
                for o in outputs_names
            ]

            if (
                any("kpt_output" in name for name in layer_names)
                and self.yolo_version != YOLOVersion.P
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
                and self.yolo_version != YOLOVersion.P
            ):
                mode = self._SEG_MODE
                # Get the segmentation outputs
                (
                    masks_outputs_values,
                    protos_output,
                    protos_len,
                ) = self._get_segmentation_outputs(output)
            else:
                mode = self._DET_MODE

            if (
                len(outputs_values[0].shape) == 4
                and outputs_values[0].shape[-1] == outputs_values[1].shape[-1]
            ):
                # RVC4
                outputs_values = [o.transpose((0, 3, 1, 2)) for o in outputs_values]
                if mode == self._SEG_MODE:
                    (
                        protos_output,
                        protos_len,
                        masks_outputs_values,
                    ) = self._reshape_seg_outputs(
                        protos_output, protos_len, masks_outputs_values
                    )

            # Get the model's input shape
            strides = (
                [8, 16, 32]
                if self.yolo_version
                not in [YOLOVersion.V3UT, YOLOVersion.V3T, YOLOVersion.V4T]
                else [16, 32]
            )
            input_shape = tuple(
                dim * strides[0] for dim in outputs_values[0].shape[2:4]
            )

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
                yolo_version=self.yolo_version,
            )

            bboxes, labels, scores, additional_output = [], [], [], []
            final_mask = np.full(input_shape, -1, dtype=float)
            for i in range(results.shape[0]):
                bbox, conf, label, other = (
                    results[i, :4],
                    results[i, 4],
                    results[i, 5].astype(int),
                    results[i, 6:],
                )

                bbox = xyxy_to_xywh_norm(bbox, input_shape)
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

            if mode == self._KPTS_MODE:
                detections_message = create_detection_message(
                    bboxes=np.array(bboxes),
                    scores=np.array(scores),
                    labels=np.array(labels),
                    keypoints=np.array(additional_output),
                )
            elif mode == self._SEG_MODE:
                detections_message = create_detection_message(
                    bboxes=np.array(bboxes),
                    scores=np.array(scores),
                    labels=np.array(labels),
                    masks=final_mask,
                )
            else:
                detections_message = create_detection_message(
                    bboxes=np.array(bboxes),
                    scores=np.array(scores),
                    labels=np.array(labels),
                )

            detections_message.setTimestamp(output.getTimestamp())

            self.out.send(detections_message)
