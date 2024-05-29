import depthai as dai
import numpy as np
import cv2

from .utils import unnormalize_image
from .utils.message_creation import create_image_msg


class ZeroDCEParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """
        Postprocessing logic for Zero-DCE model.

        Returns:
            dai.ImgFrame: uint8, HWC, BGR image represeniting the light-enhanced image.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output = output.getTensor("87")  # numpy.ndarray of shape (1, 3, 400, 600)

            image = output[0]
            image = unnormalize_image(image)

            image_message = create_image_msg(
                image=image,
                is_hwc=False,
                is_bgr=False,
            )

            self.out.send(image_message)
