from typing import Optional, Tuple

import depthai as dai
import numpy as np

from ..messages.creators import create_sam_message
from .utils.fastsam import (
    box_prompt,
    decode_fastsam_output,
    point_prompt,
    process_single_mask,
)
from .yolo import YOLOExtendedParser


class FastSAMParser(YOLOExtendedParser):
    """Parser class for parsing the output of the FastSAM model.

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
        Non-maximum suppression threshold.
    mask_conf : float
        Mask confidence threshold.
    input_shape : Tuple[int, int]
        Shape of the input image.
    prompt : str
        Prompt type.
    points : Tuple[int, int]
        Points.
    point_label : int
        Point label.
    bbox : Tuple[int, int, int, int]
        Bounding box.

    Output Message/s
    ----------------
    **Type**: SegmentationMasks

    **Description**: SegmentationMasks message containing the resulting segmentation masks given the prompt.

    Error Handling
    --------------
    """

    def __init__(
        self,
        conf_threshold: int = 0.5,
        n_classes: int = 1,
        iou_threshold: int = 0.5,
        mask_conf: float = 0.5,
        input_shape: Tuple[int, int] = (640, 640),
        prompt: str = "everything",
        points: Optional[Tuple[int, int]] = None,
        point_label: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Initialize the FastSAMParser node.

        @param conf_threshold: The confidence threshold for the detections
        @type conf_threshold: float
        @param n_classes: The number of classes in the model
        @type n_classes: int
        @param iou_threshold: The intersection over union threshold
        @type iou_threshold: float
        @param mask_conf: The mask confidence threshold
        @type mask_conf: float
        @param input_shape: The shape of the input image
        @type input_shape: Tuple[int, int]
        @param prompt: The prompt type
        @type prompt: str
        @param points: The points
        @type points: Optional[Tuple[int, int]]
        @param point_label: The point label
        @type point_label: Optional[int]
        @param bbox: The bounding box
        @type bbox: Optional[Tuple[int, int, int, int]]
        """
        YOLOExtendedParser.__init__(
            self, conf_threshold, n_classes, iou_threshold, mask_conf
        )
        self.input_shape = input_shape
        self.prompt = prompt
        self.points = points
        self.point_label = point_label
        self.bbox = bbox

    def setInputImageSize(self, width, height):
        """Sets the input image size.

        @param width: The width of the input image
        @type width: int
        @param height: The height of the input image
        @type height: int
        """
        self.input_shape = (width, height)

    def setPrompt(self, prompt):
        """Sets the prompt type.

        @param prompt: The prompt type
        @type prompt: str
        """
        self.prompt = prompt

    def setPoints(self, points):
        """Sets the points.

        @param points: The points
        @type points: Tuple[int, int]
        """
        self.points = points

    def setPointLabel(self, point_label):
        """Sets the point label.

        @param point_label: The point label
        @type point_label: int
        """
        self.point_label = point_label

    def setBoundingBox(self, bbox):
        """Sets the bounding box.

        @param bbox: The bounding box
        @type bbox: Tuple[int, int, int, int]
        """
        self.bbox = bbox

    def run(self):
        if self.prompt not in ["everything", "bbox", "point"]:
            raise ValueError("Prompt must be one of 'everything', 'bbox', or 'point'")

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
                (
                    protos_output,
                    protos_len,
                    masks_outputs_values,
                ) = self._reshape_seg_outputs(
                    protos_output, protos_len, masks_outputs_values
                )

            # Decode the outputs
            results = decode_fastsam_output(
                outputs_values,
                [8, 16, 32],
                [None, None, None],
                img_shape=self.input_shape[::-1],
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                num_classes=self.n_classes,
            )

            bboxes, masks = [], []
            for i in range(results.shape[0]):
                bbox, conf, label, seg_coeff = (
                    results[i, :4].astype(int),
                    results[i, 4],
                    results[i, 5].astype(int),
                    results[i, 6:].astype(int),
                )
                bboxes.append(bbox.tolist() + [conf, int(label)])
                hi, ai, xi, yi = seg_coeff
                mask_coeff = masks_outputs_values[hi][
                    0, ai * protos_len : (ai + 1) * protos_len, yi, xi
                ]
                mask = process_single_mask(
                    protos_output[0], mask_coeff, self.mask_conf, self.input_shape, bbox
                )
                masks.append(mask)

            results_bboxes = np.array(bboxes)
            results_masks = np.array(masks)

            if self.prompt == "bbox":
                results_masks = box_prompt(
                    results_masks, bbox=self.bbox, orig_shape=self.input_shape[::-1]
                )
            elif self.prompt == "point":
                results_masks = point_prompt(
                    results_bboxes,
                    results_masks,
                    points=self.points,
                    pointlabel=self.point_label,
                    orig_shape=self.input_shape[::-1],
                )

            segmentation_message = create_sam_message(results_masks)
            segmentation_message.setTimestamp(output.getTimestamp())

            self.out.send(segmentation_message)
