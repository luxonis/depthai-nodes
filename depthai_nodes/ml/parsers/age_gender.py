import depthai as dai

from ..messages.creators import create_age_gender_message


class AgeGenderParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the Age-Gender regression model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.

    Output Message/s
    ----------------
    **Type**: AgeGender

    **Description**: Message containing the detected person age and Classfications object for storing information about the detected person's gender.
    """

    def __init__(self):
        """Initializes the AgeGenderParser node."""
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            age = output.getTensor("age_conv3", dequantize=True).item()
            age *= 100  # convert to years
            prob = output.getTensor("prob", dequantize=True).flatten().tolist()

            age_gender_message = create_age_gender_message(age=age, gender_prob=prob)
            age_gender_message.setTimestamp(output.getTimestamp())

            self.out.send(age_gender_message)
