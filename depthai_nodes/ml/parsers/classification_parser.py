import depthai as dai
import numpy as np

from ..messages.creators import create_classification_message


class ClassificationParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Classification model.

    Parameters
    ----------
    classes : list
        List of class labels.
    is_softmax : bool = True
        True, if output is already softmaxed.

    Returns
    -------
        ClassificationMessage: A dai.Buffer object with atribute `sortedClasses` of classes and scores.
    """

    def __init__(self, classes: list, is_softmax: bool = True):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()
        self.classes = np.array(classes)
        self.n_classes = len(classes)
        self.is_softmax = is_softmax

    def setClasses(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

    def run(self) -> dai.Buffer:
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

            scores = output.getTensor(output_layer_names[0])
            scores = np.array(scores).flatten()

            if not self.is_softmax:
                ex = np.exp(scores)
                scores = ex / np.sum(ex)

            if len(scores) != self.n_classes and self.n_classes != 0:
                raise ValueError(
                    f"Number of labels and scores mismatch. Provided {self.n_classes} labels and {len(scores)} scores."
                )

            msg = create_classification_message(scores, self.classes)

            self.out.send(msg)
