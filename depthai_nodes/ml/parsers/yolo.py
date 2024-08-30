import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils.yolo import decode_yolo_output, parse_kpts, process_single_mask


class YOLOExtendedParser(dai.node.ThreadedHostNode):
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

    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints or masks and protos of the detected objects.
    """

    _KPTS_MODE = 0
    _SEG_MODE = 1

    def __init__(
        self,
        conf_threshold: int = 0.5,
        n_classes: int = 1,
        iou_threshold: int = 0.5,
        mask_conf: float = 0.5,
        n_keypoints: int = 17,
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
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.conf_threshold = conf_threshold
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.mask_conf = mask_conf
        self.n_keypoints = n_keypoints

    def setConfidenceThreshold(self, threshold):
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        self.conf_threshold = threshold

    def setNumClasses(self, n_classes):
        """Sets the number of classes in the model.

        @param numClasses: The number of classes in the model.
        @type numClasses: int
        """
        self.n_classes = n_classes

    def setIouThreshold(self, iou_threshold):
        """Sets the intersection over union threshold.

        @param iou_threshold: The intersection over union threshold.
        @type iou_threshold: float
        """
        self.iou_threshold = iou_threshold

    def setMaskConfidence(self, mask_conf):
        """Sets the mask confidence threshold.

        @param mask_conf: The mask confidence threshold.
        @type mask_conf: float
        """
        self.mask_conf = mask_conf

    def setNumKeypoints(self, n_keypoints):
        """Sets the number of keypoints in the model.

        @param n_keypoints: The number of keypoints in the model.
        @type n_keypoints: int
        """
        self.n_keypoints = n_keypoints

    def _get_segmentation_outputs(self, output):
        """Get the segmentation outputs from the Neural Network data."""
        # Get all the layer names
        layer_names = output.getAllLayerNames()
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

    def _reshape_seg_outputs(self, protos_output, protos_len, masks_outputs_values):
        """Reshape the segmentation outputs."""
        protos_output = protos_output.transpose((2, 0, 1))[np.newaxis, ...]
        protos_len = protos_output.shape[1]
        masks_outputs_values = [
            o.transpose((2, 0, 1))[np.newaxis, ...] for o in masks_outputs_values
        ]
        return protos_output, protos_len, masks_outputs_values

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped, no more data
            # Get all the layer names
            layer_names = output.getAllLayerNames()

            outputs_names = sorted([name for name in layer_names if "_yolo" in name])
            outputs_values = [
                output.getTensor(o, dequantize=True).astype(np.float32)
                for o in outputs_names
            ]

            if any("kpt_output" in name for name in layer_names):
                mode = self._KPTS_MODE
                # Get the keypoint outputs
                kpts_output_names = sorted(
                    [name for name in layer_names if "kpt_output" in name]
                )
                kpts_outputs = [
                    output.getTensor(o, dequantize=True).astype(np.float32)
                    for o in kpts_output_names
                ]
            elif any("_masks" in name for name in layer_names):
                mode = self._SEG_MODE
                # Get the segmentation outputs
                (
                    masks_outputs_values,
                    protos_output,
                    protos_len,
                ) = self._get_segmentation_outputs(output)

            if len(outputs_values[0].shape) != 4:
                # RVC4
                outputs_values = [
                    o.transpose((2, 0, 1))[np.newaxis, ...] for o in outputs_values
                ]
                if mode == self._KPTS_MODE:
                    kpts_outputs = [o[np.newaxis, ...] for o in kpts_outputs]
                elif mode == self._SEG_MODE:
                    (
                        protos_output,
                        protos_len,
                        masks_outputs_values,
                    ) = self._reshape_seg_outputs(
                        protos_output, protos_len, masks_outputs_values
                    )

            # Decode the outputs
            results = decode_yolo_output(
                outputs_values,
                [8, 16, 32],
                [None, None, None],
                kpts=kpts_outputs if mode == self._KPTS_MODE else None,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                num_classes=self.n_classes,
            )

            bboxes, labels, scores, additional_output = [], [], [], []
            for i in range(results.shape[0]):
                bbox, conf, label, other = (
                    results[i, :4].astype(int),
                    results[i, 4],
                    results[i, 5].astype(int),
                    results[i, 6:],
                )

                bboxes.append(bbox)
                labels.append(int(label))
                scores.append(conf)

                if mode == self._KPTS_MODE:
                    kpts = parse_kpts(other, self.n_keypoints)
                    additional_output.append(kpts)
                elif mode == self._SEG_MODE:
                    seg_coeff = other.astype(int)
                    hi, ai, xi, yi = seg_coeff
                    mask_coeff = masks_outputs_values[hi][
                        0, ai * protos_len : (ai + 1) * protos_len, yi, xi
                    ]
                    mask = process_single_mask(
                        protos_output[0], mask_coeff, self.mask_conf
                    )
                    additional_output.append(mask)

            if mode == self._KPTS_MODE:
                detections_message = create_detection_message(
                    np.array(bboxes),
                    np.array(scores),
                    labels,
                    keypoints=additional_output,
                )
            elif mode == self._SEG_MODE:
                detections_message = create_detection_message(
                    np.array(bboxes), np.array(scores), labels, masks=additional_output
                )
            detections_message.setTimestamp(output.getTimestamp())

            self.out.send(detections_message)
