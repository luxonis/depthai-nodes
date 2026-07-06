from dataclasses import dataclass
from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.masks_utils import (
    get_segmentation_outputs,
)
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    compute_yolo_detections,
    resolve_yolo_strides,
)


@dataclass(frozen=True)
class YOLOComputeInputs:
    subtype: YOLOSubtype
    layer_names: list[str]
    outputs_values: list[np.ndarray]
    conf_threshold: float
    n_classes: int
    iou_threshold: float
    max_det: int
    anchors: list[list[list[float]]] | np.ndarray | None
    n_keypoints: int
    label_names: list[str] | None
    keypoint_label_names: list[str] | None
    keypoint_edges: list[tuple[int, int]] | None
    input_shape: tuple[int, int] | None
    kpts_outputs: list[np.ndarray] | None
    masks_outputs_values: list[np.ndarray] | None
    protos_output: np.ndarray | None
    protos_len: int | None
    mask_conf: float
    v26_mask_coeffs: np.ndarray | None
    v26_protos: np.ndarray | None
    v26_pose_kpts: np.ndarray | None


class YOLOExtendedParser(BaseParser):
    """Parser class for parsing the output of the YOLO Instance Segmentation and Pose
    Estimation models.

    Attributes
    ----------
    conf_threshold : float
        Confidence score threshold for detected faces.
    n_classes : int
        Number of classes in the model.
    label_names : list[str] | None
        Names of the classes.
    iou_threshold : float
        Intersection over union threshold.
    mask_conf : float
        Mask confidence threshold.
    n_keypoints : int
        Number of keypoints in the model.
    anchors : list[list[list[float]]] | None
        Anchors for the YOLO model (optional).
    strides : list[int] | tuple[int, ...] | None
        Strides for the YOLO output heads.
    keypoint_label_names : list[str] | None
        Labels for the keypoints.
    keypoint_edges : list[tuple[int, int]] | None
        Keypoint connection pairs for visualizing the skeleton. Example: [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is connected to keypoint 2, etc.
    subtype : str
        Version of the YOLO model.


    Output Message/s
    ----------------
    **Type**: dai.ImgDetections

    **Description**: dai.ImgDetections message containing bounding boxes, labels, label names, confidence scores, and keypoints or masks and protos of the detected objects.
    """

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(
        self,
        conf_threshold: float = 0.5,
        n_classes: int = 1,
        label_names: list[str] | None = None,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        n_keypoints: int = 17,
        max_det: int = 300,
        anchors: list[list[list[float]]] | None = None,
        subtype: str = "",
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
    ):
        """Initialize the parser node.

        @param conf_threshold: The confidence threshold for the detections
        @type conf_threshold: float
        @param n_classes: The number of classes in the model
        @type n_classes: int
        @param label_names: The names of the classes
        @type label_names: list[str] | None
        @param iou_threshold: The intersection over union threshold
        @type iou_threshold: float
        @param mask_conf: The mask confidence threshold
        @type mask_conf: float
        @param n_keypoints: The number of keypoints in the model
        @type n_keypoints: int
        @param anchors: The anchors for the YOLO model
        @type anchors: list[list[list[float]]] | None
        @param subtype: The version of the YOLO model
        @type subtype: str
        @param keypoint_label_names: The labels for the keypoints
        @type keypoint_label_names: list[str] | None
        @param keypoint_edges: Connection pairs of the keypoints. Example: [(0,1),
            (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1,
            keypoint 1 is connected to keypoint 2, etc.
        @type keypoint_edges: list[tuple[int, int]] | None
        """
        super().__init__()

        self.output_layer_names = []
        self.conf_threshold = conf_threshold
        self.n_classes = n_classes
        self.label_names = label_names
        self.iou_threshold = iou_threshold
        self.mask_conf = mask_conf
        self.n_keypoints = n_keypoints
        self.max_det = max_det
        self.anchors = anchors
        self.strides: list[int] | tuple[int, ...] | None = None
        self.keypoint_label_names = keypoint_label_names
        self.keypoint_edges = keypoint_edges
        self.input_shape = None
        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO version {subtype}. Supported YOLO versions are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

        self._logger.debug(
            f"YOLOExtendedParser initialized with conf_threshold={self.conf_threshold}, n_classes={self.n_classes}, label_names={self.label_names}, iou_threshold={self.iou_threshold}, mask_conf={self.mask_conf}, n_keypoints={self.n_keypoints}, max_det={self.max_det}, anchors={self.anchors}, strides={self.strides}, subtype='{self.subtype}', keypoint_label_names={self.keypoint_label_names}, keypoint_edges={self.keypoint_edges}"
        )

    def setOutputLayerNames(self, output_layer_names: list[str]) -> None:
        """Sets the output layer names for the parser.

        @param output_layer_names: The output layer names for the parser.
        @type output_layer_names: list[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(layer, str) for layer in output_layer_names):
            raise ValueError("Output layer names must be a list of strings.")

        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence score threshold must be a float.")

        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence score threshold must be between 0 and 1.")

        self.conf_threshold = threshold
        self._logger.debug(f"Confidence score threshold set to {self.conf_threshold}")

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
        self._logger.debug(f"Number of classes set to {self.n_classes}")

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
        self._logger.debug(
            f"Intersection over union threshold set to {self.iou_threshold}"
        )

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
        self._logger.debug(f"Mask confidence threshold set to {self.mask_conf}")

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
        self._logger.debug(f"Number of keypoints set to {self.n_keypoints}")

    def setAnchors(self, anchors: list[list[list[float]]]) -> None:
        """Sets the anchors for the YOLO model.

        @param anchors: The anchors for the YOLO model.
        @type anchors: list[list[list[float]]]
        """
        for anchor in anchors:
            if not isinstance(anchor, list):
                raise ValueError("Anchors must be a list of lists.")
            if not all(isinstance(val, list) for val in anchor):
                raise ValueError("Anchors must be a list of lists of lists.")
            if not all(isinstance(val, float) for sublist in anchor for val in sublist):
                raise ValueError("Anchors must be a list of lists of floats.")

        self.anchors = anchors
        self._logger.debug(f"Anchors set to {self.anchors}")

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

        self._logger.debug(f"Subtype set to {self.subtype}")

    def setLabelNames(self, label_names: list[str]) -> None:
        """Sets the names of the classes.

        @param label_names: The names of the classes.
        @type label_names: list[str]
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Label names must be a list of strings.")

        self.label_names = label_names
        self._logger.debug(f"Label names set to {self.label_names}")

    def setKeypointLabelNames(self, keypoint_label_names: list[str]) -> None:
        """Sets the label names for the keypoints.

        @param keypoint_label_names: The labels for the keypoints.
        @type keypoint_label_names: list[str]
        """
        if not isinstance(keypoint_label_names, list):
            raise ValueError("Keypoint labels must be a list.")
        if not all(isinstance(label, str) for label in keypoint_label_names):
            raise ValueError("Keypoint labels must be a list of strings.")

        self.keypoint_label_names = keypoint_label_names
        self._logger.debug(f"Keypoint label names set to {self.keypoint_label_names}")

    def setKeypointEdges(self, keypoint_edges: list[tuple[int, int]]) -> None:
        """Sets the edges for the keypoints.

        @param keypoint_edges: The edges for the keypoints.
        @type keypoint_edges: list[tuple[int, int]]
        """
        if not isinstance(keypoint_edges, list) and not isinstance(
            keypoint_edges, tuple
        ):
            raise ValueError("Keypoint edges must be a list or tuple.")
        if not all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(i, int) for i in edge)
            for edge in keypoint_edges
        ):
            raise ValueError("Keypoint edges must be a list of tuples of integers.")

        self.keypoint_edges = keypoint_edges
        self._logger.debug(f"Keypoint edges set to {self.keypoint_edges}")

    def build(
        self,
        head_config: dict[str, Any],
    ):
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: YOLOExtendedParser
        """

        output_layers = head_config.get("outputs", [])
        subtype = head_config.get("subtype", self.subtype)
        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

        if self.subtype == YOLOSubtype.V26:
            # YOLO26 end2end: task type is inferred from output layer names at runtime.
            # The detection tensor is already decoded and has shape (N, A, 5+nc):
            # [x1, y1, x2, y2, conf, cls_0, ..., cls_nc-1].
            # Auxiliary outputs depend on the task:
            # - Detection: 'output_yolo26'
            # - Segmentation: 'output_yolo26' plus 'output_masks' (N, A, nm) and
            #   'protos_output' (N, nm, proto_h, proto_w)
            # - Pose: 'output_yolo26' plus 'kpt_output' (N, A, nk)
            kps_layer_names = [name for name in output_layers if "kpt_output" in name]
            masks_layer_names = [
                name for name in output_layers if "output_masks" in name
            ]
            protos_layer_names = [name for name in output_layers if "protos" in name]

            if kps_layer_names:
                bbox_layer_names = [
                    name for name in output_layers if name == "output_yolo26"
                ]
                if len(bbox_layer_names) != 1 or len(kps_layer_names) != 1:
                    raise ValueError(
                        "YOLO26 pose requires 2 outputs: 'output_yolo26' and 'kpt_output'."
                    )
            elif masks_layer_names:
                bbox_layer_names = [
                    name for name in output_layers if name == "output_yolo26"
                ]
                self._protos_layer_name = (
                    protos_layer_names[0] if protos_layer_names else "protos_output"
                )
                if len(bbox_layer_names) != 1 or len(masks_layer_names) != 1:
                    raise ValueError(
                        "YOLO26 segmentation requires 3 outputs: 'output_yolo26', 'output_masks', and 'protos_output'."
                    )
                masks_layer_names = masks_layer_names + protos_layer_names
            else:
                bbox_layer_names = list(output_layers)
                if len(bbox_layer_names) != 1:
                    raise ValueError(
                        "YOLO26 detection requires a single output layer with shape (N, A, 5+nc)."
                    )
        else:
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
        self.strides = head_config.get("strides", self.strides)
        self.n_keypoints = head_config.get("n_keypoints", self.n_keypoints)
        self.max_det = head_config.get("max_det", self.max_det)
        self.label_names = head_config.get("classes", self.label_names)
        self.keypoint_label_names = head_config.get(
            "keypoint_label_names", self.keypoint_label_names
        )
        keypoint_edges = head_config.get("skeleton_edges", self.keypoint_edges)
        if keypoint_edges:
            self.keypoint_edges = [tuple(edge) for edge in keypoint_edges]
        if self.subtype == YOLOSubtype.V26:
            # For YOLO26 end2end we no longer have FPN outputs to infer input size,
            # so we rely on model_inputs from the NN archive.
            inputs = head_config.get("model_inputs", [])
            if inputs:
                input_shape = inputs[0].get("shape")
                input_layout = inputs[0].get("layout")
                if input_shape and input_layout:
                    if input_layout == "NCHW":
                        self.input_shape = (input_shape[2], input_shape[3])
                    elif input_layout == "NHWC":
                        self.input_shape = (input_shape[1], input_shape[2])
            # Get n_prototypes for segmentation mode
            if any("output_masks" in name for name in output_layers):
                self.n_prototypes = head_config.get("n_prototypes", 32)

        self._logger.debug(
            f"YOLOExtendedParser built with conf_threshold={self.conf_threshold}, n_classes={self.n_classes}, label_names={self.label_names}, iou_threshold={self.iou_threshold}, mask_conf={self.mask_conf}, n_keypoints={self.n_keypoints}, anchors={self.anchors}, strides={self.strides}, subtype='{self.subtype}', keypoint_label_names={self.keypoint_label_names}, keypoint_edges={self.keypoint_edges}"
        )

        return self

    def run(self):
        self._logger.debug("YOLOExtendedParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data
            extracted = self.extract(output)
            payload = self.compute(extracted)
            self.emit(output, payload)

    def extract(self, output: dai.NNData) -> YOLOComputeInputs:
        layer_names = self.output_layer_names or output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {layer_names}")

        kpts_outputs = None
        masks_outputs_values = None
        protos_output = None
        protos_len = None
        v26_mask_coeffs = None
        v26_protos = None
        v26_pose_kpts = None

        if self.subtype == YOLOSubtype.V26:
            if any("output_masks" in name for name in layer_names):
                outputs_values = [
                    output.getTensor("output_yolo26", dequantize=True).astype(
                        np.float32
                    )
                ]
                v26_mask_coeffs = output.getTensor(
                    "output_masks", dequantize=True
                ).astype(np.float32)
                v26_protos = output.getTensor(
                    self._protos_layer_name,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)[0]
            elif any("kpt_output" in name for name in layer_names):
                outputs_values = [
                    output.getTensor("output_yolo26", dequantize=True).astype(
                        np.float32
                    )
                ]
                v26_pose_kpts = output.getTensor("kpt_output", dequantize=True).astype(
                    np.float32
                )
            else:
                outputs_values = [
                    output.getTensor(o, dequantize=True).astype(np.float32)
                    for o in list(layer_names)
                ]
        else:
            outputs_names = sorted(
                [name for name in layer_names if "_yolo" in name or "yolo-" in name]
            )
            outputs_values = [
                output.getTensor(
                    o,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)
                for o in outputs_names
            ]

            if (
                any("kpt_output" in name for name in layer_names)
                and self.subtype != YOLOSubtype.P
            ):
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
                (
                    masks_outputs_values,
                    protos_output,
                    protos_len,
                ) = get_segmentation_outputs(output)

        if self.subtype == YOLOSubtype.V26:
            if self.input_shape is None:
                raise ValueError(
                    "YOLO26 parsing requires model input shape in head_config."
                )
            input_shape = self.input_shape
        else:
            strides = resolve_yolo_strides(
                self.strides,
                self.subtype,
                num_outputs=len(outputs_values),
            )
            input_shape = tuple(
                dim * strides[0] for dim in outputs_values[0].shape[2:4]
            )

        return YOLOComputeInputs(
            subtype=self.subtype,
            layer_names=list(layer_names),
            outputs_values=outputs_values,
            conf_threshold=self.conf_threshold,
            n_classes=self.n_classes,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            anchors=self.anchors,
            n_keypoints=self.n_keypoints,
            label_names=self.label_names,
            keypoint_label_names=self.keypoint_label_names,
            keypoint_edges=self.keypoint_edges,
            input_shape=input_shape,
            kpts_outputs=kpts_outputs,
            masks_outputs_values=masks_outputs_values,
            protos_output=protos_output,
            protos_len=protos_len,
            mask_conf=self.mask_conf,
            v26_mask_coeffs=v26_mask_coeffs,
            v26_protos=v26_protos,
            v26_pose_kpts=v26_pose_kpts,
        )

    @staticmethod
    def compute(inputs: YOLOComputeInputs) -> dict[str, Any]:
        return compute_yolo_detections(
            subtype=inputs.subtype,
            layer_names=inputs.layer_names,
            outputs_values=inputs.outputs_values,
            conf_threshold=inputs.conf_threshold,
            n_classes=inputs.n_classes,
            iou_threshold=inputs.iou_threshold,
            max_det=inputs.max_det,
            anchors=inputs.anchors,
            n_keypoints=inputs.n_keypoints,
            label_names=inputs.label_names,
            keypoint_label_names=inputs.keypoint_label_names,
            keypoint_edges=inputs.keypoint_edges,
            input_shape=inputs.input_shape,
            kpts_outputs=inputs.kpts_outputs,
            masks_outputs_values=inputs.masks_outputs_values,
            protos_output=inputs.protos_output,
            protos_len=inputs.protos_len,
            mask_conf=inputs.mask_conf,
            v26_mask_coeffs=inputs.v26_mask_coeffs,
            v26_protos=inputs.v26_protos,
            v26_pose_kpts=inputs.v26_pose_kpts,
        )

    def emit(self, output: dai.NNData, payload: dict[str, Any]) -> None:
        mode = payload["mode"]
        if mode == self._KPTS_MODE:
            detections_message = create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                keypoints=payload["keypoints"],
                keypoints_scores=payload["keypoints_scores"],
                keypoint_label_names=payload["keypoint_label_names"],
                keypoint_edges=payload["keypoint_edges"],
            )
        elif mode == self._SEG_MODE:
            detections_message = create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                masks=payload["masks"],
            )
        else:
            detections_message = create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
            )

        detections_message.setTimestamp(output.getTimestamp())
        detections_message.setTimestampDevice(output.getTimestampDevice())
        detections_message.setSequenceNum(output.getSequenceNum())
        transformation = output.getTransformation()
        if transformation is not None:
            detections_message.setTransformation(transformation)

        self._logger.debug(
            f"Created detection message with {len(payload['bboxes'])} detections"
        )
        self.out.send(detections_message)
        self._logger.debug("Detection message sent successfully")
