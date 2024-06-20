import depthai as dai

from .utils import unnormalize_image
from ..messages.creation_functions import create_image_message


class ImageOutputParser(dai.node.ThreadedHostNode):
    def __init__(self, output_is_bgr=False):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.output_is_bgr = output_is_bgr

    def setBGROutput(self):
        self.output_is_bgr = True

    def run(self):
        """
        Postprocessing logic for image-to-image models (e.g. DnCNN3, zero-dce etc.).

        Returns:
            dai.ImgFrame: uint8, grayscale HW / colorscale HWC BGR image.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            if len(output_layer_names) != 1:
                raise ValueError(
                    f"Expected 1 output layer, got {len(output_layer_names)}."
                )
            output = output.getTensor(output_layer_names[0])

            if len(output.shape) == 4:
                image = output[0]
            elif len(output.shape) == 3:
                image = output
            else:
                raise ValueError(
                    f"Expected 3- or 4-dimensional output, got {len(output.shape)}-dimensional",
                )

            image = unnormalize_image(image)

            image_message = create_image_message(
                image=image,
                is_bgr=self.output_is_bgr,
            )

            self.out.send(image_message)
