import depthai as dai

from ..messages.creators import create_image_message
from .utils import unnormalize_image


class ImageOutputParser(dai.node.ThreadedHostNode):
    """Parser class for image-to-image models (e.g. DnCNN3, zero-dce etc.) where the
    output is a modifed image (denoised, enhanced etc.).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_is_bgr : bool
        Flag indicating if the output image is in BGR (Blue-Green-Red) format.

    Output Message/s
    -------
    **Type**: dai.ImgFrame

    **Description**: Image message containing the output image e.g. denoised or enhanced images.

    Error Handling
    --------------
    **ValueError**: If the output is not 3- or 4-dimensional.

    **ValueError**: If the number of output layers is not 1.
    """

    def __init__(self, output_is_bgr=False):
        """Initializes ImageOutputParser node.

        @param output_is_bgr: Flag indicating if the output image is in BGR.
        @type output_is_bgr: bool
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.output_is_bgr = output_is_bgr

    def setBGROutput(self):
        """Sets the flag indicating that output image is in BGR."""
        self.output_is_bgr = True

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            if len(output_layer_names) != 1:
                raise ValueError(
                    f"Expected 1 output layer, got {len(output_layer_names)}."
                )

            output_image = output.getTensor(output_layer_names[0], dequantize=True)

            if len(output_image.shape) == 4:
                image = output_image[0]
            elif len(output_image.shape) == 3:
                image = output_image
            else:
                raise ValueError(
                    f"Expected 3- or 4-dimensional output, got {len(output_image.shape)}-dimensional",
                )

            image = unnormalize_image(image)

            image_message = create_image_message(
                image=image,
                is_bgr=self.output_is_bgr,
            )
            image_message.setTimestamp(output.getTimestamp())

            self.out.send(image_message)
