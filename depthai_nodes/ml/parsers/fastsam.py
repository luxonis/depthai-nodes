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
from .yolo import YOLOParser


class FastSAMParser(YOLOParser):
    def __init__(
            self,
            confidence_threshold: int = 0.5,
            num_classes: int = 1,
            iou_threshold: int = 0.5,
            mask_conf: float = 0.5,
            input_shape: Tuple[int, int] = (640, 640),
            prompt: str = "everything",
            points: Optional[Tuple[int, int]] = None,
            point_label: Optional[int] = None,
            bbox: Optional[Tuple[int, int, int, int]] = None
        ):
        """Initialize the YOLOParser node.

        @param confidence_threshold: The confidence threshold for the detections
        @type confidence_threshold: float
        @param num_classes: The number of classes in the model
        @type num_classes: int
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
        YOLOParser.__init__(self, confidence_threshold, num_classes, iou_threshold, mask_conf)
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
        while self.isRunning():
            try:
                nnDataIn : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break # Pipeline was stopped, no more data
            # Get all the layer names
            layer_names = nnDataIn.getAllLayerNames()

            outputs_names = sorted([name for name in layer_names if "_yolo" in name])
            outputs_values = [nnDataIn.getTensor(o, dequantize=True).astype(np.float32) for o in outputs_names]
            # Get the segmentation outputs
            masks_outputs_values, protos_output, protos_len = self._get_segmentation_outputs(nnDataIn)

            if len(outputs_values[0].shape) != 4:
                # RVC4
                outputs_values = [o.transpose((2, 0, 1))[np.newaxis, ...] for o in outputs_values]
                protos_output, protos_len, masks_outputs_values = self._reshape_seg_outputs(protos_output, protos_len, masks_outputs_values)

            # Decode the outputs
            results = decode_fastsam_output(
                outputs_values,
                [8, 16, 32],
                [None, None, None],
                img_shape=self.input_shape[::-1],
                conf_thres=self.confidence_threshold,
                iou_thres=self.iou_threshold,
                num_classes=self.num_classes
            )

            bboxes, masks = [], []
            for i in range(results.shape[0]):
                bbox, conf, label, seg_coeff = results[i, :4].astype(int), results[i, 4], results[i, 5].astype(int), results[i, 6:].astype(int)
                bboxes.append(bbox.tolist() + [conf, int(label)])
                hi, ai, xi, yi = seg_coeff
                mask_coeff = masks_outputs_values[hi][0, ai*protos_len:(ai+1)*protos_len, yi, xi]
                mask = process_single_mask(protos_output[0], mask_coeff, self.mask_conf, self.input_shape, bbox)
                masks.append(mask)

            results_bboxes = np.array(bboxes)
            results_masks = np.array(masks)

            if self.prompt == "bbox":
                results_masks = box_prompt(results_masks, bbox=self.bbox, orig_shape=self.input_shape[::-1])
            elif self.prompt == "point":
                results_masks = point_prompt(results_bboxes, results_masks, points=self.points, pointlabel=self.point_label, orig_shape=self.input_shape[::-1])

            segmentation_message = create_sam_message(results_masks)
            self.out.send(segmentation_message)
