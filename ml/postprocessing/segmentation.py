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
            segmentation_mask = np.vstack((np.zeros((1, segmentation_mask.shape[1], segmentation_mask.shape[2]), dtype=np.float32), segmentation_mask))
            segmentation_mask[segmentation_mask < self.threshold] = 0
            overlay_image = np.argmax(segmentation_mask, axis=0).reshape(segmentation_mask.shape[1], segmentation_mask.shape[2], 1).astype(np.uint8)

            imgFrame = create_segmentation_message(overlay_image)
            self.out.send(imgFrame)