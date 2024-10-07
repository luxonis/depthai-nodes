from typing import List

import depthai as dai
import numpy as np

from ..messages.creators import (
    create_classification_message,
    create_multi_classification_message,
)


class ClassificationParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Classification model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    classes : List[str]
        List of class names to be used for linking with their respective scores. Expected to be in the same order as Neural Network's output. If not provided, the message will only return sorted scores.
    is_softmax : bool = True
        If False, the scores are converted to probabilities using softmax function.
    n_classes : int = len(classes)
        Number of provided classes. This variable is set automatically based on provided classes.

    Output Message/s
    ----------------
    **Type** : Classifications(dai.Buffer):
         An object with attributes `classes` and `scores`. `classes` is a list of classes, sorted in descending order of scores. `scores` is a list of corresponding scores.
    """

    def __init__(self, classes: List[str] = None, is_softmax: bool = True):
        """Initializes the ClassificationParser node.

        @param classes: List of class names to be used for linking with their respective
            scores.
        @param is_softmax: If False, the scores are converted to probabilities using
            softmax function.
        """

        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()
        self.classes = classes if classes is not None else []
        self.n_classes = len(self.classes)
        self.is_softmax = is_softmax

    def setClasses(self, classes: List[str]):
        """Sets the class names for the classification model.

        @param classes: List of class names to be used for linking with their respective
            scores.
        """
        self.classes = classes if classes is not None else []
        self.n_classes = len(self.classes)

    def setSoftmax(self, is_softmax: bool):
        """Sets the softmax flag for the classification model.

        @param is_softmax: If False, the parser will convert the scores to probabilities
            using softmax function.
        """
        self.is_softmax = is_softmax

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

            if self.n_classes == 0:
                raise ValueError("Classes must be provided for classification.")

            scores = output.getTensor(output_layer_names[0], dequantize=True).astype(
                np.float32
            )
            scores = np.array(scores).flatten()

            if len(scores) != self.n_classes and self.n_classes != 0:
                raise ValueError(
                    f"Number of labels and scores mismatch. Provided {self.n_classes} class names and {len(scores)} scores."
                )

            if not self.is_softmax:
                ex = np.exp(scores)
                scores = ex / np.sum(ex)

            msg = create_classification_message(self.classes, scores)
            msg.setTimestamp(output.getTimestamp())

            self.out.send(msg)


class MultiClassificationParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Multiple Classification model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    classification_attributes : List[str]
        List of attributes to be classified.
    classification_labels : List[List[str]]
        List of class labels for each attribute in `classification_attributes`

    Output Message/s
    ----------------
    **Type**: CompositeMessage

    **Description**: A CompositeMessage containing a dictionary of classification attributes as keys and their respective Classifications as values.
    """

    def __init__(
        self,
        classification_attributes: List[str] = None,
        classification_labels: List[List[str]] = None,
    ):
        """Initializes the MultipleClassificationParser node."""
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()
        self.classification_attributes: List[str] = classification_attributes
        self.classification_labels: List[List[str]] = classification_labels

    def setClassificationAttributes(self, classification_attributes: List[str]):
        """Sets the classification attributes for the multiple classification model.

        @param classification_attributes: List of attributes to be classified.
        @type classification_attributes: List[str]
        """
        self.classification_attributes = classification_attributes

    def setClassificationLabels(self, classification_labels: List[List[str]]):
        """Sets the classification labels for the multiple classification model.

        @param classification_labels: List of class labels for each attribute.
        @type classification_labels: List[List[str]]
        """
        self.classification_labels = classification_labels

    def run(self):
        if not self.classification_attributes:
            raise ValueError("Classification attributes must be provided.")
        if not self.classification_labels:
            raise ValueError("Classification labels must be provided.")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            layer_names = output.getAllLayerNames()

            scores = []
            for layer_name in layer_names:
                scores.append(
                    output.getTensor(layer_name, dequantize=True).flatten().tolist()
                )

            multi_classification_message = create_multi_classification_message(
                self.classification_attributes, scores, self.classification_labels
            )
            multi_classification_message.setTimestamp(output.getTimestamp())

            self.out.send(multi_classification_message)
