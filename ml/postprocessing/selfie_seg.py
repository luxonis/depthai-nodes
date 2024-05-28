import depthai as dai
import numpy as np
import cv2
from .utils.message_creation.depth_segmentation import create_depth_segmentation_msg

class SeflieSegParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        threshold=0.5,
        input_size=(256, 144),
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.input_size = input_size
        self.threshold = threshold

    def setConfidenceThreshold(self, threshold):
        self.threshold = threshold

    def setInputSize(self, width, height):
        self.input_size = (width, height)

    def run(self):
        """
        Postprocessing logic for SCRFD model.

        Returns:
            ...
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

            imgFrame = create_depth_segmentation_msg(overlay_image, 'raw8')
            self.out.send(imgFrame)