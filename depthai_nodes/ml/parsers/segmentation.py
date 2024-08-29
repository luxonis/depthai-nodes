import depthai as dai
import numpy as np

from ..messages.creators import create_segmentation_message


class SegmentationParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the segmentation models.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    background_class : bool
        Whether to add additional layer for background.

    Output Message/s
    ----------------
    **Type**: dai.ImgFrame

    **Description**: Segmentation message containing the segmentation mask. Every pixel belongs to exactly one class.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.

    **ValueError**: If the number of dimensions of the output tensor is not E{3}.
    """

    def __init__(self, background_class=False):
        """Initializes the SegmentationParser node.

        @param background_class: Whether to add additional layer for background.
        @type background_class: bool
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()
        self.background_class = background_class

    def setBackgroundClass(self, background_class):
        """Sets the background class.

        @param background_class: Whether to add additional layer for background.
        @type background_class: bool
        """
        self.background_class = background_class

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

            segmentation_mask = output.getTensor(output_layer_names[0], dequantize=True)
            if len(segmentation_mask.shape) == 4:
                segmentation_mask = segmentation_mask[0]
            else:
                segmentation_mask = segmentation_mask.transpose(2, 0, 1)

            if len(segmentation_mask.shape) != 3:
                raise ValueError(
                    f"Expected 3D output tensor, got {len(segmentation_mask.shape)}D."
                )

            if self.background_class:
                segmentation_mask = np.vstack(
                    (
                        np.zeros(
                            (1, segmentation_mask.shape[1], segmentation_mask.shape[2]),
                            dtype=np.float32,
                        ),
                        segmentation_mask,
                    )
                )

            class_map = (
                np.argmax(segmentation_mask, axis=0)
                .reshape(segmentation_mask.shape[1], segmentation_mask.shape[2], 1)
                .astype(np.uint8)
            )

            imgFrame = create_segmentation_message(class_map)
            imgFrame.setTimestamp(output.getTimestamp())
            self.out.send(imgFrame)
