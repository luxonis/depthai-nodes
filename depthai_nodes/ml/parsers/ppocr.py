from typing import List

import depthai as dai
import numpy as np

from ..messages.creators import create_classification_sequence_message
from .classification import ClassificationParser


class PaddleOCRParser(ClassificationParser):
    """Postprocessing logic for PaddleOCR text recognition model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    characters: List[str]
        List of available characters for the text recognition model.
    ignored_indexes: List[int]
        List of indexes to ignore during classification generation (e.g., background class, blank space).
    remove_duplicates: bool
        If True, removes consecutive duplicates from the sequence.
    concatenate_text: bool
        If True, concatenates consecutive words based on the predicted spaces.
    is_softmax: bool
        If False, the scores are converted to probabilities using softmax function.

    Output Message/s
    ----------------
    **Type**: Classifications(dai.Buffer)

    **Description**: An object with attributes `classes` and `scores`. `classes` is a list containing the predicted text. `scores` is a list of corresponding probability scores.

    See also
    --------
    Official PaddleOCR repository:
    https://github.com/PaddlePaddle/PaddleOCR
    """

    def __init__(
        self,
        characters: List[str] = None,
        ignored_indexes: List[int] = None,
        remove_duplicates: bool = False,
        concatenate_text: bool = True,
        is_softmax: bool = True,
    ):
        """Initializes the PaddleOCR Parser node.

        @param characters: List of available characters for the text recognition model.
        @type characters: List[str]
        @param ignored_indexes: List of indexes to ignore during classification
            generation (e.g., background class, blank space).
        @type ignored_indexes: List[int]
        @param remove_duplicates: If True, removes consecutive duplicates from the
            sequence.
        @type remove_duplicates: bool
        @param concatenate_text: If True, concatenates consecutive words based on the
            predicted spaces.
        @type concatenate_text: bool
        @param is_softmax: If False, the scores are converted to probabilities using
            softmax function.
        """
        super().__init__(characters, is_softmax)
        self.ignored_indexes = [0] if ignored_indexes is None else ignored_indexes
        self.remove_duplicates = remove_duplicates
        self.concatenate_text = concatenate_text

    def setRemoveDuplicates(self, remove_duplicates: bool):
        """Sets the remove_duplicates flag for the classification sequence model.

        @param remove_duplicates: If True, removes consecutive duplicates from the
            sequence.
        """
        self.remove_duplicates = remove_duplicates

    def setIgnoredIndexes(self, ignored_indexes: List[int]):
        """Sets the ignored_indexes for the classification sequence model.

        @param ignored_indexes: A list of indexes to ignore during classification
            generation.
        """
        self.ignored_indexes = ignored_indexes

    def setConcatenateText(self, concatenate_text: bool):
        """Sets the concatenate_text flag for the classification sequence model.

        @param concatenate_text: If True, concatenates consecutive words based on
            predicted spaces.
        """
        self.concatenate_text = concatenate_text

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()

            except dai.MessageQueue.QueueException:
                break

            output_layer_names = output.getAllLayerNames()
            if len(output_layer_names) != 1:
                raise ValueError(
                    f"Expected 1 output layer, got {len(output_layer_names)}."
                )

            if self.n_classes == 0:
                raise ValueError("Classes must be provided for classification.")

            if any([len(ch) > 1 for ch in self.classes]):
                raise ValueError("Each character should only be a single character.")

            scores = output.getTensor(output_layer_names[0], dequantize=True).astype(
                np.float32
            )

            if len(scores.shape) != 3:
                raise ValueError(f"Scores should be a 3D array, got {scores.shape}.")

            if scores.shape[0] == 1:
                scores = scores[0]
            elif scores.shape[2] == 1:
                scores = scores[:, :, 0]
            else:
                raise ValueError(
                    "Scores should be a 3D array of shape (1, sequence_length, n_classes) or (sequence_length, n_classes, 1)."
                )

            if not self.is_softmax:
                scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

            msg = create_classification_sequence_message(
                classes=self.classes,
                scores=scores,
                remove_duplicates=self.remove_duplicates,
                ignored_indexes=self.ignored_indexes,
                concatenate_text=self.concatenate_text,
            )
            msg.setTimestamp(output.getTimestamp())

            self.out.send(msg)
