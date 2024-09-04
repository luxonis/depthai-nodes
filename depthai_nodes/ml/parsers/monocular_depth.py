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
    depth_limit : float
        The maximum depth value (in meters) to be used in the depth map.

    Output Message/s
    ----------------
    **Type**: dai.ImgFrame

    **Description**: Depth message containing the depth map. The depth map is represented with dai.ImgFrame.

    Error Handling
    --------------
    **ValueError**: If the number of output layers is not E{1}.
    """

    def __init__(self, depth_type="relative", depth_limit=0.0):
        """Initializes the MonocularDepthParser node.

        @param depth_type: Type of depth output (relative or metric).
        @type depth_type: Literal['relative', 'metric']
        @param depth_limit: The maximum depth value (in meters) to be used in the depth
            map. The default value is 0, which means no limit.
        @type depth_limit: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.depth_type = depth_type
        self.depth_limit = depth_limit

    def setRelativeDepthType(self):
        """Sets the depth type to relative."""
        self.depth_type = "relative"

    def setMetricDepthType(self):
        """Sets the depth type to metric."""
        self.depth_type = "metric"

    def setDepthLimit(self, depth_limit):
        """Sets the depth limit."""
        self.depth_limit = depth_limit

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

            output_map = output.getTensor(output_layer_names[0], dequantize=True)

            if len(output_map.shape) == 3:
                if output_map.shape[0] == 1:
                    depth_map = output_map[0]
                elif output_map.shape[2] == 1:
                    depth_map = output_map[:, :, 0]
            elif len(output_map.shape) == 2:
                depth_map = output_map
            elif len(output_map.shape) == 4:
                depth_map = output_map[0][0]
            else:
                raise ValueError(
                    f"Expected 3- or 2-dimensional output, got {len(output_map.shape)}-dimensional",
                )

            depth_message = create_depth_message(
                depth_map=depth_map,
                depth_type=self.depth_type,
                depth_limit=self.depth_limit,
            )
            depth_message.setTimestamp(output.getTimestamp())
            self.out.send(depth_message)
