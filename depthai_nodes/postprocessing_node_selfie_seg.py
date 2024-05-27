import depthai as dai
import numpy as np
import cv2

class SeflieSegParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        threshold=0.5,
        input_size=(256, 144),
        mask_color=[0, 255, 0],
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.input_size = input_size
        self.threshold = threshold
        self.mask_color = mask_color

    def setMaskColor(self, mask_color):
        self.mask_color = mask_color

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
            overlay_image = np.ones((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8) * 255
            overlay_image[segmentation_mask] = self.mask_color

            imgFrame = dai.ImgFrame()
            imgFrame.setFrame(overlay_image)
            imgFrame.setWidth(overlay_image.shape[1])
            imgFrame.setHeight(overlay_image.shape[0])
            imgFrame.setType(dai.ImgFrame.Type.BGR888i)

            self.out.send(imgFrame)