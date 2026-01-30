from typing import Any, Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import (
    normalize_bboxes,
    xyxy_to_xywh,
)
from depthai_nodes.node.parsers.utils.masks_utils import (
    get_segmentation_outputs,
    process_single_mask,
)
from depthai_nodes.node.parsers.utils.yolo import (
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
    label_names : Optional[List[str]]
        Names of the classes.
    iou_threshold : float
        Intersection over union threshold.
    mask_conf : float
        Mask confidence threshold.
    n_keypoints : int
        Number of keypoints in the model.
    anchors : Optional[List[List[List[float]]]]
        Anchors for the YOLO model (optional).
    keypoint_label_names : Optional[List[str]]
        Labels for the keypoints.
    keypoint_edges : Optional[List[Tuple[int, int]]]
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
    _OBB_MODE = 3

    def __init__(
        self,
        conf_threshold: float = 0.5,
        n_classes: int = 1,
        label_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        n_keypoints: int = 17,
        max_det: int = 300,
        anchors: Optional[List[List[List[float]]]] = None,
        subtype: str = "",
        keypoint_label_names: Optional[List[str]] = None,
        keypoint_edges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize the parser node.

        @param conf_threshold: The confidence threshold for the detections
        @type conf_threshold: float
        @param n_classes: The number of classes in the model
        @type n_classes: int
        @param label_names: The names of the classes
        @type label_names: Optional[List[str]]
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
        @param keypoint_label_names: The labels for the keypoints
        @type keypoint_label_names: Optional[List[str]]
        @param keypoint_edges: Connection pairs of the keypoints. Example: [(0,1),
            (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1,
            keypoint 1 is connected to keypoint 2, etc.
        @type keypoint_edges: Optional[List[Tuple[int, int]]]
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
            f"YOLOExtendedParser initialized with conf_threshold={self.conf_threshold}, n_classes={self.n_classes}, label_names={self.label_names}, iou_threshold={self.iou_threshold}, mask_conf={self.mask_conf}, n_keypoints={self.n_keypoints}, max_det={self.max_det}, anchors={self.anchors}, subtype='{self.subtype}', keypoint_label_names={self.keypoint_label_names}, keypoint_edges={self.keypoint_edges}"
        )

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

    def setLabelNames(self, label_names: List[str]) -> None:
        """Sets the names of the classes.

        @param label_names: The names of the classes.
        @type label_names: List[str]
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Label names must be a list of strings.")

        self.label_names = label_names
        self._logger.debug(f"Label names set to {self.label_names}")

    def setKeypointLabelNames(self, keypoint_label_names: List[str]) -> None:
        """Sets the label names for the keypoints.

        @param keypoint_label_names: The labels for the keypoints.
        @type keypoint_label_names: List[str]
        """
        if not isinstance(keypoint_label_names, list):
            raise ValueError("Keypoint labels must be a list.")
        if not all(isinstance(label, str) for label in keypoint_label_names):
            raise ValueError("Keypoint labels must be a list of strings.")

        self.keypoint_label_names = keypoint_label_names
        self._logger.debug(f"Keypoint label names set to {self.keypoint_label_names}")

    def setKeypointEdges(self, keypoint_edges: List[Tuple[int, int]]) -> None:
        """Sets the edges for the keypoints.

        @param keypoint_edges: The edges for the keypoints.
        @type keypoint_edges: List[Tuple[int, int]]
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

    def _parse_v26_pose_kpts(
        self,
        kpts: np.ndarray,
        n_keypoints: int,
        img_shape: Tuple[int, int],
    ) -> List[Tuple[float, float, float]]:
        """Parse keypoints from YOLO26-POSE output.

        YOLO26-POSE keypoints are already decoded in pixel coordinates.
        Format: [x1, y1, v1, x2, y2, v2, ...] where x,y are in pixels
        and v is visibility score (already sigmoided).

        @param kpts: Keypoint values for a single detection.
        @type kpts: np.ndarray
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        @param img_shape: Image shape (height, width).
        @type img_shape: Tuple[int, int]
        @return: List of (x, y, visibility) tuples normalized to [0, 1].
        @rtype: List[Tuple[float, float, float]]
        """
        h, w = img_shape
        kps = []
        ndim = len(kpts) // n_keypoints
        for idx in range(0, len(kpts), ndim):
            # Keypoints are in pixel coordinates, normalize to [0, 1]
            x = kpts[idx] / w
            y = kpts[idx + 1] / h
            conf = kpts[idx + 2] if ndim == 3 else 1.0
            kps.append((x, y, conf))
        return kps

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
        subtype = head_config.get("subtype", self.subtype)
        try:
            self.subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

        if self.subtype == YOLOSubtype.V26:
            # YOLO26 end2end export provides a single output (N, A, 4+nc) without FPN heads.
            bbox_layer_names = list(output_layers)
            kps_layer_names = []
            masks_layer_names = []
            if len(bbox_layer_names) != 1:
                raise ValueError(
                    "YOLO26 requires a single output layer with shape (N, A, 4+nc)."
                )
        elif self.subtype == YOLOSubtype.V26_SEG:
            # YOLO26-SEG end2end export provides 3 outputs:
            # - output: (N, A, 4+nc) detection output
            # - mask_output: (N, A, nm) mask coefficients
            # - protos_output: (N, nm, H, W) prototype masks
            bbox_layer_names = [name for name in output_layers if name == "output"]
            masks_layer_names = [name for name in output_layers if "mask_output" in name]
            protos_layer_names = [name for name in output_layers if "protos" in name]
            kps_layer_names = []
            self._protos_layer_name = protos_layer_names[0] if protos_layer_names else "protos_output"
            if len(bbox_layer_names) != 1 or len(masks_layer_names) != 1:
                raise ValueError(
                    "YOLO26-SEG requires 3 outputs: 'output', 'mask_output', and 'protos_output'."
                )
            # Include protos in output_layer_names for proper layer retrieval
            masks_layer_names = masks_layer_names + protos_layer_names
        elif self.subtype == YOLOSubtype.V26_POSE:
            # YOLO26-POSE end2end export provides 2 outputs:
            # - output: (N, A, 4+nc) detection output
            # - kpt_output: (N, A, nk) decoded keypoints in pixel coordinates
            bbox_layer_names = [name for name in output_layers if name == "output"]
            kps_layer_names = [name for name in output_layers if name == "kpt_output"]
            masks_layer_names = []
            if len(bbox_layer_names) != 1 or len(kps_layer_names) != 1:
                raise ValueError(
                    "YOLO26-POSE requires 2 outputs: 'output' and 'kpt_output'."
                )
        elif self.subtype == YOLOSubtype.V26_OBB:
            # YOLO26-OBB end2end export provides 2 outputs:
            # - output: (N, A, 4+nc) detection output (xywh format)
            # - angle_output: (N, A, 1) rotation angles in radians
            bbox_layer_names = [name for name in output_layers if name == "output"]
            angle_layer_names = [
                name for name in output_layers if name == "angle_output"
            ]
            kps_layer_names = []
            masks_layer_names = []
            if len(bbox_layer_names) != 1 or len(angle_layer_names) != 1:
                raise ValueError(
                    "YOLO26-OBB requires 2 outputs: 'output' and 'angle_output'."
                )
            self.output_layer_names = bbox_layer_names + angle_layer_names
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
        self.n_keypoints = head_config.get("n_keypoints", self.n_keypoints)
        self.max_det = head_config.get("max_det", self.max_det)
        self.label_names = head_config.get("classes", self.label_names)
        self.keypoint_label_names = head_config.get(
            "keypoint_label_names", self.keypoint_label_names
        )
        keypoint_edges = head_config.get("skeleton_edges", self.keypoint_edges)
        if keypoint_edges:
            self.keypoint_edges = [tuple(edge) for edge in keypoint_edges]
        if self.subtype in (YOLOSubtype.V26, YOLOSubtype.V26_SEG, YOLOSubtype.V26_POSE, YOLOSubtype.V26_OBB):
            # For YOLO26 variants end2end we no longer have FPN outputs to infer input size,
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
            # Get n_prototypes for V26_SEG
            if self.subtype == YOLOSubtype.V26_SEG:
                self.n_prototypes = head_config.get("n_prototypes", 32)

        self._logger.debug(
            f"YOLOExtendedParser built with conf_threshold={self.conf_threshold}, n_classes={self.n_classes}, label_names={self.label_names}, iou_threshold={self.iou_threshold}, mask_conf={self.mask_conf}, n_keypoints={self.n_keypoints}, anchors={self.anchors}, subtype='{self.subtype}', keypoint_label_names={self.keypoint_label_names}, keypoint_edges={self.keypoint_edges}"
        )

        return self

    def run(self):
        self._logger.debug("YOLOExtendedParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data
            # Get all the layer names
            layer_names = self.output_layer_names or output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layer_names}")

            if self.subtype == YOLOSubtype.V26:
                # YOLO26 end2end output is a single NCD tensor: (N, A, 4+nc).
                outputs_names = list(layer_names)
                outputs_values = [
                    output.getTensor(o, dequantize=True).astype(np.float32)
                    for o in outputs_names
                ]
            elif self.subtype == YOLOSubtype.V26_SEG:
                # YOLO26-SEG end2end outputs:
                # - output: (N, A, 4+nc) detection output
                # - mask_output: (N, A, nm) mask coefficients
                # - protos_output: (N, nm, H, W) prototype masks
                outputs_names = ["output"]
                outputs_values = [
                    output.getTensor("output", dequantize=True).astype(np.float32)
                ]
                # Get mask coefficients and protos separately
                self._mask_coeffs = output.getTensor(
                    "mask_output", dequantize=True
                ).astype(np.float32)
                self._protos = output.getTensor(
                    self._protos_layer_name, dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW
                ).astype(np.float32)
            elif self.subtype == YOLOSubtype.V26_POSE:
                # YOLO26-POSE end2end outputs:
                # - output: (N, A, 4+nc) detection output
                # - kpt_output: (N, A, nk) decoded keypoints in pixel coordinates
                outputs_names = ["output"]
                outputs_values = [
                    output.getTensor("output", dequantize=True).astype(np.float32)
                ]
                # Get keypoints separately (already decoded in pixel coordinates)
                self._kpts_output = output.getTensor(
                    "kpt_output", dequantize=True
                ).astype(np.float32)
            elif self.subtype == YOLOSubtype.V26_OBB:
                # YOLO26-OBB end2end outputs:
                # - output: (N, A, 4+nc) detection output (xywh format)
                # - angle_output: (N, A, 1) rotation angles in radians
                outputs_names = ["output"]
                outputs_values = [
                    output.getTensor("output", dequantize=True).astype(np.float32)
                ]
                # Get angles separately
                self._angle_output = output.getTensor(
                    "angle_output", dequantize=True
                ).astype(np.float32)
            else:
                outputs_names = sorted(
                    [name for name in layer_names if "_yolo" in name or "yolo-" in name]
                )
                outputs_values = [
                    output.getTensor(
                        o, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
                    ).astype(np.float32)
                    for o in outputs_names
                ]

            if self.subtype == YOLOSubtype.V26_POSE:
                # V26_POSE is always keypoints mode
                mode = self._KPTS_MODE
            elif self.subtype == YOLOSubtype.V26_OBB:
                # V26_OBB is always OBB mode
                mode = self._OBB_MODE
            elif (
                any("kpt_output" in name for name in layer_names)
                and self.subtype != YOLOSubtype.P
                and self.subtype not in (YOLOSubtype.V26, YOLOSubtype.V26_SEG)
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
            elif self.subtype == YOLOSubtype.V26_SEG:
                # V26_SEG is always segmentation mode
                mode = self._SEG_MODE
            elif (
                any("_masks" in name for name in layer_names)
                and self.subtype != YOLOSubtype.P
                and self.subtype != YOLOSubtype.V26
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
            if self.subtype in (YOLOSubtype.V26, YOLOSubtype.V26_SEG, YOLOSubtype.V26_POSE, YOLOSubtype.V26_OBB):
                if self.input_shape is None:
                    raise ValueError(
                        "YOLO26 variants parsing requires model input shape in head_config."
                    )
                input_shape = self.input_shape
            else:
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
            if self.subtype not in (YOLOSubtype.V26, YOLOSubtype.V26_SEG, YOLOSubtype.V26_POSE, YOLOSubtype.V26_OBB) and self.anchors is not None:
                self.anchors = np.array(self.anchors).reshape(len(strides), -1)

            # Ensure the number of classes is correct
            if self.subtype not in (YOLOSubtype.V26, YOLOSubtype.V26_SEG, YOLOSubtype.V26_POSE, YOLOSubtype.V26_OBB):
                num_classes_check = (
                    outputs_values[0].shape[1] - 5
                    if self.anchors is None
                    else (outputs_values[0].shape[1] // self.anchors.shape[0]) - 5
                )
                if num_classes_check != self.n_classes:
                    raise ValueError(
                        f"The provided number of classes {self.n_classes} does not match the model's {num_classes_check}."
                    )

            # Ensure the number of keypoints is correct
            if mode == self._KPTS_MODE:
                num_keypoints_check = kpts_outputs[0].shape[1] // 3
                if num_keypoints_check != self.n_keypoints:
                    raise ValueError(
                        f"The provided number of keypoints {self.n_keypoints} does not match the model's {num_keypoints_check}."
                    )

            # Decode the outputs
            if self.subtype == YOLOSubtype.V26:
                if len(outputs_values) != 1:
                    raise ValueError("YOLO26 requires a single output layer.")
                raw = outputs_values[0]
                if raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26 output must be 3D (N, A, 4+nc). Got shape {raw.shape}."
                    )
                expected_last_dim = 4 + self.n_classes
                if raw.shape[-1] != expected_last_dim and raw.shape[1] == expected_last_dim:
                    # NCD layout with D=4+nc stored as (N, 4+nc, A)
                    raw = np.transpose(raw, (0, 2, 1))
                if raw.shape[-1] != expected_last_dim:
                    raise ValueError(
                        f"YOLO26 output last dim must be 4+nc. Got shape {raw.shape}."
                    )
                # YOLO26 end2end output is already decoded (xyxy in pixels) but not top-k filtered.
                # Conf threshold and topK happens here
                results = raw[0]
                if results.size:
                    boxes = results[:, :4]
                    scores = results[:, 4:]
                    cls_ids = scores.argmax(axis=-1).astype(np.float32)
                    cls_scores = scores.max(axis=-1)
                    keep = cls_scores >= self.conf_threshold
                    if np.any(keep):
                        boxes = boxes[keep]
                        cls_scores = cls_scores[keep]
                        cls_ids = cls_ids[keep]
                        k = min(self.max_det, cls_scores.shape[0])
                        if cls_scores.shape[0] > k:
                            topk_idx = np.argpartition(-cls_scores, k - 1)[:k]
                            order = np.argsort(-cls_scores[topk_idx])
                            topk_idx = topk_idx[order]
                        else:
                            topk_idx = np.argsort(-cls_scores)
                        boxes = boxes[topk_idx]
                        cls_scores = cls_scores[topk_idx]
                        cls_ids = cls_ids[topk_idx]
                        results = np.concatenate(
                            [boxes, cls_scores[:, None], cls_ids[:, None]], axis=1
                        ).astype(np.float32)
                    else:
                        results = np.zeros((0, 6), dtype=np.float32)
                else:
                    results = np.zeros((0, 6), dtype=np.float32)
            elif self.subtype == YOLOSubtype.V26_SEG:
                # YOLO26-SEG end2end decoding with mask coefficients
                if len(outputs_values) != 1:
                    raise ValueError("YOLO26-SEG requires detection output layer.")
                raw = outputs_values[0]
                mask_coeffs_raw = self._mask_coeffs
                protos = self._protos[0]  # (nm, H, W)

                if raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-SEG detection output must be 3D (N, A, 4+nc). Got shape {raw.shape}."
                    )
                expected_last_dim = 4 + self.n_classes
                if raw.shape[-1] != expected_last_dim and raw.shape[1] == expected_last_dim:
                    raw = np.transpose(raw, (0, 2, 1))
                if raw.shape[-1] != expected_last_dim:
                    raise ValueError(
                        f"YOLO26-SEG detection output last dim must be 4+nc. Got shape {raw.shape}."
                    )

                # Handle mask_coeffs layout: should be (N, A, nm)
                if mask_coeffs_raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-SEG mask_coeffs must be 3D (N, A, nm). Got shape {mask_coeffs_raw.shape}."
                    )
                if mask_coeffs_raw.shape[1] != raw.shape[1] and mask_coeffs_raw.shape[2] == raw.shape[1]:
                    # Layout is (N, nm, A), transpose to (N, A, nm)
                    mask_coeffs_raw = np.transpose(mask_coeffs_raw, (0, 2, 1))

                det_results = raw[0]  # (A, 4+nc)
                mask_coeffs_all = mask_coeffs_raw[0]  # (A, nm)
                kept_mask_coeffs = []

                if det_results.size:
                    boxes = det_results[:, :4]
                    scores = det_results[:, 4:]
                    cls_ids = scores.argmax(axis=-1).astype(np.float32)
                    cls_scores = scores.max(axis=-1)
                    keep = cls_scores >= self.conf_threshold

                    if np.any(keep):
                        # Get indices of kept detections for mask retrieval
                        keep_indices = np.where(keep)[0]
                        boxes = boxes[keep]
                        cls_scores = cls_scores[keep]
                        cls_ids = cls_ids[keep]
                        mask_coeffs_kept = mask_coeffs_all[keep_indices]

                        k = min(self.max_det, cls_scores.shape[0])
                        if cls_scores.shape[0] > k:
                            topk_idx = np.argpartition(-cls_scores, k - 1)[:k]
                            order = np.argsort(-cls_scores[topk_idx])
                            topk_idx = topk_idx[order]
                        else:
                            topk_idx = np.argsort(-cls_scores)

                        boxes = boxes[topk_idx]
                        cls_scores = cls_scores[topk_idx]
                        cls_ids = cls_ids[topk_idx]
                        kept_mask_coeffs = mask_coeffs_kept[topk_idx]

                        results = np.concatenate(
                            [boxes, cls_scores[:, None], cls_ids[:, None]], axis=1
                        ).astype(np.float32)
                    else:
                        results = np.zeros((0, 6), dtype=np.float32)
                        kept_mask_coeffs = np.zeros((0, mask_coeffs_all.shape[1]), dtype=np.float32)
                else:
                    results = np.zeros((0, 6), dtype=np.float32)
                    kept_mask_coeffs = np.zeros((0, mask_coeffs_all.shape[1]), dtype=np.float32)

                # Store for SEG_MODE processing
                self._v26_seg_mask_coeffs = kept_mask_coeffs
                self._v26_seg_protos = protos
            elif self.subtype == YOLOSubtype.V26_POSE:
                # YOLO26-POSE end2end decoding with keypoints
                if len(outputs_values) != 1:
                    raise ValueError("YOLO26-POSE requires detection output layer.")
                raw = outputs_values[0]
                kpts_raw = self._kpts_output

                if raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-POSE detection output must be 3D (N, A, 4+nc). Got shape {raw.shape}."
                    )
                expected_last_dim = 4 + self.n_classes
                if raw.shape[-1] != expected_last_dim and raw.shape[1] == expected_last_dim:
                    raw = np.transpose(raw, (0, 2, 1))
                if raw.shape[-1] != expected_last_dim:
                    raise ValueError(
                        f"YOLO26-POSE detection output last dim must be 4+nc. Got shape {raw.shape}."
                    )

                # Handle kpts layout: should be (N, A, nk)
                if kpts_raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-POSE kpts must be 3D (N, A, nk). Got shape {kpts_raw.shape}."
                    )
                expected_nk = self.n_keypoints * 3
                if kpts_raw.shape[-1] != expected_nk and kpts_raw.shape[1] == expected_nk:
                    # Layout is (N, nk, A), transpose to (N, A, nk)
                    kpts_raw = np.transpose(kpts_raw, (0, 2, 1))

                det_results = raw[0]  # (A, 4+nc)
                kpts_all = kpts_raw[0]  # (A, nk)
                kept_kpts = []

                if det_results.size:
                    boxes = det_results[:, :4]
                    scores = det_results[:, 4:]
                    cls_ids = scores.argmax(axis=-1).astype(np.float32)
                    cls_scores = scores.max(axis=-1)
                    keep = cls_scores >= self.conf_threshold

                    if np.any(keep):
                        keep_indices = np.where(keep)[0]
                        boxes = boxes[keep]
                        cls_scores = cls_scores[keep]
                        cls_ids = cls_ids[keep]
                        kpts_kept = kpts_all[keep_indices]

                        k = min(self.max_det, cls_scores.shape[0])
                        if cls_scores.shape[0] > k:
                            topk_idx = np.argpartition(-cls_scores, k - 1)[:k]
                            order = np.argsort(-cls_scores[topk_idx])
                            topk_idx = topk_idx[order]
                        else:
                            topk_idx = np.argsort(-cls_scores)

                        boxes = boxes[topk_idx]
                        cls_scores = cls_scores[topk_idx]
                        cls_ids = cls_ids[topk_idx]
                        kept_kpts = kpts_kept[topk_idx]

                        results = np.concatenate(
                            [boxes, cls_scores[:, None], cls_ids[:, None]], axis=1
                        ).astype(np.float32)
                    else:
                        results = np.zeros((0, 6), dtype=np.float32)
                        kept_kpts = np.zeros((0, kpts_all.shape[1]), dtype=np.float32)
                else:
                    results = np.zeros((0, 6), dtype=np.float32)
                    kept_kpts = np.zeros((0, kpts_all.shape[1]), dtype=np.float32)

                # Store for KPTS_MODE processing
                self._v26_pose_kpts = kept_kpts
            elif self.subtype == YOLOSubtype.V26_OBB:
                # YOLO26-OBB end2end decoding with angles
                if len(outputs_values) != 1:
                    raise ValueError("YOLO26-OBB requires detection output layer.")
                raw = outputs_values[0]
                angles_raw = self._angle_output

                if raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-OBB detection output must be 3D (N, A, 4+nc). Got shape {raw.shape}."
                    )
                expected_last_dim = 4 + self.n_classes
                if raw.shape[-1] != expected_last_dim and raw.shape[1] == expected_last_dim:
                    raw = np.transpose(raw, (0, 2, 1))
                if raw.shape[-1] != expected_last_dim:
                    raise ValueError(
                        f"YOLO26-OBB detection output last dim must be 4+nc. Got shape {raw.shape}."
                    )

                # Handle angles layout: should be (N, A, 1)
                if angles_raw.ndim != 3:
                    raise ValueError(
                        f"YOLO26-OBB angles must be 3D (N, A, 1). Got shape {angles_raw.shape}."
                    )
                if angles_raw.shape[-1] != 1 and angles_raw.shape[1] == 1:
                    # Layout is (N, 1, A), transpose to (N, A, 1)
                    angles_raw = np.transpose(angles_raw, (0, 2, 1))

                det_results = raw[0]  # (A, 4+nc)
                angles_all = angles_raw[0]  # (A, 1)
                kept_angles = []

                if det_results.size:
                    # Note: OBB boxes are in xywh format (center x, center y, width, height)
                    boxes = det_results[:, :4]
                    scores = det_results[:, 4:]
                    cls_ids = scores.argmax(axis=-1).astype(np.float32)
                    cls_scores = scores.max(axis=-1)
                    keep = cls_scores >= self.conf_threshold

                    if np.any(keep):
                        keep_indices = np.where(keep)[0]
                        boxes = boxes[keep]
                        cls_scores = cls_scores[keep]
                        cls_ids = cls_ids[keep]
                        angles_kept = angles_all[keep_indices]

                        k = min(self.max_det, cls_scores.shape[0])
                        if cls_scores.shape[0] > k:
                            topk_idx = np.argpartition(-cls_scores, k - 1)[:k]
                            order = np.argsort(-cls_scores[topk_idx])
                            topk_idx = topk_idx[order]
                        else:
                            topk_idx = np.argsort(-cls_scores)

                        boxes = boxes[topk_idx]
                        cls_scores = cls_scores[topk_idx]
                        cls_ids = cls_ids[topk_idx]
                        kept_angles = angles_kept[topk_idx]

                        results = np.concatenate(
                            [boxes, cls_scores[:, None], cls_ids[:, None]], axis=1
                        ).astype(np.float32)
                    else:
                        results = np.zeros((0, 6), dtype=np.float32)
                        kept_angles = np.zeros((0, 1), dtype=np.float32)
                else:
                    results = np.zeros((0, 6), dtype=np.float32)
                    kept_angles = np.zeros((0, 1), dtype=np.float32)

                # Store for OBB_MODE processing
                self._v26_obb_angles = kept_angles
            else:
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

            bboxes, labels, label_names, scores, additional_output = [], [], [], [], []
            final_mask = np.full(input_shape, 255, dtype=np.uint8)
            for i in range(results.shape[0]):
                bbox, conf, label, other = (
                    results[i, :4],
                    results[i, 4],
                    results[i, 5].astype(int),
                    results[i, 6:],
                )

                if self.subtype == YOLOSubtype.V26_OBB:
                    # V26_OBB: boxes are already in xywh (center) format in pixels
                    # Just normalize to [0, 1] range
                    bbox = bbox.reshape(1, 4)
                    bbox[0, 0] /= input_shape[1]  # cx / width
                    bbox[0, 1] /= input_shape[0]  # cy / height
                    bbox[0, 2] /= input_shape[1]  # w / width
                    bbox[0, 3] /= input_shape[0]  # h / height
                    bbox = bbox[0]
                else:
                    bbox = xyxy_to_xywh(bbox.reshape(1, 4))
                    bbox = normalize_bboxes(
                        bbox, height=input_shape[0], width=input_shape[1]
                    )[0]
                bboxes.append(bbox)
                labels.append(int(label))
                if self.label_names:
                    label_names.append(self.label_names[int(label)])
                scores.append(conf)

                if mode == self._KPTS_MODE:
                    if self.subtype == YOLOSubtype.V26_POSE:
                        # V26_POSE: keypoints are already decoded in pixel coordinates
                        kpts = self._parse_v26_pose_kpts(
                            self._v26_pose_kpts[i], self.n_keypoints, input_shape
                        )
                    else:
                        kpts = parse_kpts(other, self.n_keypoints, input_shape)
                    additional_output.append(kpts)
                elif mode == self._OBB_MODE:
                    if self.subtype == YOLOSubtype.V26_OBB:
                        # V26_OBB: angles are stored separately in radians, convert to degrees
                        angle_rad = self._v26_obb_angles[i][0]  # Single angle value in radians
                        angle_deg = np.degrees(angle_rad)  # Convert to degrees for create_detection_message
                        additional_output.append(angle_deg)
                elif mode == self._SEG_MODE:
                    if self.subtype == YOLOSubtype.V26_SEG:
                        # V26_SEG: mask coefficients are directly available
                        mask_coeff = self._v26_seg_mask_coeffs[i]
                        mask = process_single_mask(
                            self._v26_seg_protos, mask_coeff, self.mask_conf, bbox
                        )
                    else:
                        # Other YOLO versions: extract mask coefficients from indexed outputs
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
                    label_names=label_names,
                    keypoints=keypoints,
                    keypoints_scores=keypoints_scores,
                    keypoint_label_names=self.keypoint_label_names,
                    keypoint_edges=self.keypoint_edges,
                )
            elif mode == self._SEG_MODE:
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                    label_names=label_names,
                    masks=final_mask,
                )
            elif mode == self._OBB_MODE:
                # OBB mode: include rotation angles
                angles = np.array(additional_output) if additional_output else np.array([])
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                    label_names=label_names,
                    angles=angles,
                )
            else:
                detections_message = create_detection_message(
                    bboxes=bboxes,
                    scores=np.array(scores),
                    labels=np.array(labels),
                    label_names=label_names,
                )

            detections_message.setTimestamp(output.getTimestamp())
            detections_message.setTimestampDevice(output.getTimestampDevice())
            detections_message.setSequenceNum(output.getSequenceNum())
            transformation = output.getTransformation()
            if transformation is not None:
                detections_message.setTransformation(transformation)

            self._logger.debug(
                f"Created detection message with {len(bboxes)} detections"
            )

            self.out.send(detections_message)

            self._logger.debug("Detection message sent successfully")
