import depthai as dai
import numpy as np

from ..messages.creators import create_classification_message


class ClassificationParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Classification model.

    Parameters
    ----------
    classes : list[str]
        List of class labels.
    is_softmax : bool = True
        True, if output is already softmaxed.

    Returns
    -------
        Classifications: dai.Buffer
            An object with parameter `classes`, which is a list of items like [class_name, probability_score].
            If no class names are provided, class_name is set to None.
    """

    def __init__(self, classes: list[str] = None, is_softmax: bool = True):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()
        if classes is None:
            self.classes = []
        else:
            self.classes = np.array(classes)
        self.n_classes = len(classes)
        self.is_softmax = is_softmax

    def setClasses(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

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

            scores = output.getTensor(output_layer_names[0])
            scores = np.array(scores).flatten()

            if len(scores) != self.n_classes and self.n_classes != 0:
                raise ValueError(
                    f"Number of labels and scores mismatch. Provided {self.n_classes} class names and {len(scores)} scores."
                )

            if not self.is_softmax:
                ex = np.exp(scores)
                scores = ex / np.sum(ex)

            msg = create_classification_message(scores, self.classes)

            self.out.send(msg)
