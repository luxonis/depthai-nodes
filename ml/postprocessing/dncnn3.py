import depthai as dai

from .utils import unnormalize_image
from .utils.message_creation import create_image_msg


class DnCNN3Parser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """
        Postprocessing logic for DnCNN3 model.

        Returns:
            dai.ImgFrame: uint8, GRAYSCALE denoised image.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output = output.getTensor("80")  # numpy.ndarray of shape (1, 1, 321, 481)

            image = output[0][0]
            image = unnormalize_image(image)

            image_message = create_image_msg(
                image=image,
                is_grayscale=True,
            )

            self.out.send(image_message)
