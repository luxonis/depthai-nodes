import depthai as dai
import numpy as np
import cv2
from .utils.message_creation import create_segmentation_message

class MPSeflieSegParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        threshold=0.5,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.threshold = threshold

    def setConfidenceThreshold(self, threshold):
        self.threshold = threshold

    def run(self):
        """
        Postprocessing logic for MediaPipe Selfie Segmentation model.

        Returns:
            Segmenation mask with two classes 1 - person, 0 - background.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
                print(f"output = {output}")
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            print(f"Layer names = {output.getAllLayerNames()}")

            segmentation_mask = output.getTensor("output")
            segmentation_mask = segmentation_mask[0].squeeze() > self.threshold
            overlay_image = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 1), dtype=np.uint8)
            overlay_image[segmentation_mask] = 1

            imgFrame = create_segmentation_message(overlay_image)
            self.out.send(imgFrame)