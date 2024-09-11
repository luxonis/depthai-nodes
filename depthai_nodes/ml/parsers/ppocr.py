from typing import List

import depthai as dai
import numpy as np

from ..messages.creators import create_classification_sequence_message
from .classification import ClassificationParser


class PaddleOCRParser(ClassificationParser):
    """"""

    def __init__(
        self,
        classes: List[str] = None,
        ignored_indexes: List[int] = None,
        remove_duplicates: bool = False,
        concatenate_text: bool = True,
        is_softmax: bool = True,
    ):
        """Initializes the PaddleOCR Parser node.

        @param classes: List of class names to be
        """
        super().__init__(classes, is_softmax)
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
