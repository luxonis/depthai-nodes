import depthai as dai

from ..messages.creators import create_depth_message


class MonocularDepthParser(dai.node.ThreadedHostNode):
    """Parser class for monocular depth models (e.g. Depth Anything model).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    depth_type : str
        Type of depth output (relative or metric).

    Output Message/s
    ----------------
    **Type**: dai.ImgFrame

    **Description**: Depth message containing the depth map. The depth map is represented with dai.ImgFrame.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.
    """

    def __init__(self, depth_type="relative"):
        """Initializes the MonocularDepthParser node.

        @param depth_type: Type of depth output (relative or metric).
        @type depth_type: str
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.depth_type = depth_type

    def setRelativeDepthType(self):
        """Sets the depth type to relative."""
        self.depth_type = "relative"

    def setMetricDepthType(self):
        """Sets the depth type to metric."""
        self.depth_type = "metric"

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
            output = output.getTensor(output_layer_names[0])

            depth_map = output[0]

            depth_message = create_depth_message(
                depth_map=depth_map,
                depth_type=self.depth_type,
            )
            depth_message.setTimestamp(output.getTimestamp())
            self.out.send(depth_message)
