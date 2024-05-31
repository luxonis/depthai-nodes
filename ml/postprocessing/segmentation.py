import depthai as dai
import numpy as np
import cv2
from .utils.message_creation import create_segmentation_message

class SegmentationParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        threshold=0.5,
        num_classes=2,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.threshold = threshold
        self.num_classes = num_classes

    def setConfidenceThreshold(self, threshold):
        self.threshold = threshold

    def setNumClasses(self, num_classes):
        self.num_classes = num_classes

    def run(self):
        """
        Postprocessing logic for Segmentation model with `num_classes` classes including background at index 0.

        Returns:
            Segmenation mask with `num_classes` classes, 0 - background.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            segmentation_mask = output.getTensor("output")
            segmentation_mask = segmentation_mask[0]  # num_clases x H x W
            overlay_image = np.zeros((segmentation_mask.shape[1], segmentation_mask.shape[2], 1), dtype=np.uint8)

            for class_id in range(self.num_classes-1):
                class_mask = segmentation_mask[class_id] > self.threshold
                overlay_image[class_mask] = class_id + 1

            imgFrame = create_segmentation_message(overlay_image)
            self.out.send(imgFrame)