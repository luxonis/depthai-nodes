import depthai as dai
import numpy as np

from ..messages.creators import create_cluster_message
from .utils.ufld import decode_ufld


class LaneDetectionParser(dai.node.ThreadedHostNode):
    """
    Parser class for Ultra-Fast-Lane-Detection model. It expects one ouput layer containing the lane detection results.
    It supports two versions of the model: CuLane and TuSimple. Results are representented with clusters of points.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    row_anchors : List[int]
        List of row anchors.
    griding_num : int
        Griding number.
    cls_num_per_lane : int
        Number of points per lane.
    input_shape : Tuple[int, int]
        Input shape.
    """

    def __init__(
        self,
        row_anchors=None,
        griding_num=None,
        cls_num_per_lane=None,
        input_shape=(288, 800),
    ):
        """Initializes the lane detection parser node.

        @param row_anchors: List of row anchors.
        @type row_anchors: List[int]
        @param griding_num: Griding number.
        @type griding_num: int
        @param cls_num_per_lane: Number of points per lane.
        @type cls_num_per_lane: int
        @param input_shape: Input shape.
        @type input_shape: Tuple[int, int]
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()
        self.row_anchors = row_anchors
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.input_shape = input_shape

    def setRowAnchors(self, row_anchors):
        """Set the row anchors for the lane detection model.

        @param row_anchors: List of row anchors.
        @type row_anchors: List[int]
        """
        self.row_anchors = row_anchors

    def setGridingNum(self, griding_num):
        """Set the griding number for the lane detection model.

        @param griding_num: Griding number.
        @type griding_num: int
        """
        self.griding_num = griding_num

    def setClsNumPerLane(self, cls_num_per_lane):
        """Set the number of points per lane for the lane detection model.

        @param cls_num_per_lane: Number of classes per lane.
        @type cls_num_per_lane: int
        """
        self.cls_num_per_lane = cls_num_per_lane

    def setInputShape(self, input_shape):
        """Set the input shape for the lane detection model.

        @param input_shape: Input shape.
        @type input_shape: Tuple[int, int]
        """
        self.input_shape = input_shape

    def run(self):
        if self.row_anchors is None:
            raise ValueError("Row anchors must be specified!")
        if self.griding_num is None:
            raise ValueError("Griding number must be specified!")
        if self.cls_num_per_lane is None:
            raise ValueError("Number of points per lane must be specified!")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            tensor = output.getFirstTensor(dequantize=True).astype(np.float32)
            y = tensor[0]

            points = decode_ufld(
                anchors=self.row_anchors,
                griding_num=self.griding_num,
                cls_num_per_lane=self.cls_num_per_lane,
                INPUT_WIDTH=self.input_shape[1],
                INPUT_HEIGHT=self.input_shape[0],
                y=y,
            )

            message = create_cluster_message(points)
            message.setTimestamp(output.getTimestamp())
            self.out.send(message)
