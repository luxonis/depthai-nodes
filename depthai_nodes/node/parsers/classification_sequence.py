from typing import Any, Dict, List

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_classification_sequence_message
from depthai_nodes.node.parsers.utils import softmax

from .classification import ClassificationParser


class ClassificationSequenceParser(ClassificationParser):
    """Postprocessing logic for a classification sequence model. The model predicts the
    classes multiple times and returns a list of predicted classes, where each item
    corresponds to the relative step in the sequence. In addition to time series
    classification, this parser can also be used for text recognition models where words
    can be interpreted as a sequence of characters (classes).

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    classes: List[str]
        List of available classes for the model.
    is_softmax: bool
        If False, the scores are converted to probabilities using softmax function.
    ignored_indexes: List[int]
        List of indexes to ignore during classification generation (e.g., background class, blank space).
    remove_duplicates: bool
        If True, removes consecutive duplicates from the sequence.
    concatenate_classes: bool
        If True, concatenates consecutive words based on the predicted spaces.

    Output Message/s
    ----------------
    **Type**: Classifications(dai.Buffer)

    **Description**:
        An object with attributes `classes` and `scores`. `classes` is a list containing the predicted classes. `scores` is a list of corresponding probability scores.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        classes: List[str] = None,
        is_softmax: bool = True,
        ignored_indexes: List[int] = None,
        remove_duplicates: bool = False,
        concatenate_classes: bool = False,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param classes: List of available classes for the model.
        @type classes: List[str]
        @param ignored_indexes: List of indexes to ignore during classification
            generation (e.g., background class, blank space).
        @type ignored_indexes: List[int]
        @param is_softmax: If False, the scores are converted to probabilities using
            softmax function.
        @type is_softmax: bool
        @param remove_duplicates: If True, removes consecutive duplicates from the
            sequence.
        @type remove_duplicates: bool
        @param concatenate_classes: If True, concatenates consecutive words based on the
            predicted spaces.
        @type concatenate_classes: bool
        """
        super().__init__(
            output_layer_name=output_layer_name, classes=classes, is_softmax=is_softmax
        )

        self.ignored_indexes = ignored_indexes if ignored_indexes is not None else []
        self.remove_duplicates = remove_duplicates
        self.concatenate_classes = concatenate_classes
        self._logger.debug(
            f"ClassificationSequenceParser initialized with output_layer_name='{output_layer_name}', classes={classes}, is_softmax={is_softmax}, ignored_indexes={ignored_indexes}, remove_duplicates={remove_duplicates}, concatenate_classes={concatenate_classes}"
        )

    def setRemoveDuplicates(self, remove_duplicates: bool) -> None:
        """Sets the remove_duplicates flag for the classification sequence model.

        @param remove_duplicates: If True, removes consecutive duplicates from the
            sequence.
        @type remove_duplicates: bool
        """
        if not isinstance(remove_duplicates, bool):
            raise ValueError("remove_duplicates must be a boolean.")
        self.remove_duplicates = remove_duplicates
        self._logger.debug(f"Remove duplicates set to {remove_duplicates}")

    def setIgnoredIndexes(self, ignored_indexes: List[int]) -> None:
        """Sets the ignored_indexes for the classification sequence model.

        @param ignored_indexes: A list of indexes to ignore during classification
            generation.
        @type ignored_indexes: List[int]
        """
        if not isinstance(ignored_indexes, list):
            raise ValueError("Ignored indexes must be a list.")
        if not all(isinstance(index, int) for index in ignored_indexes):
            raise ValueError("All ignored indexes must be integers.")
        self.ignored_indexes = ignored_indexes
        self._logger.debug(f"Ignored indexes set to {ignored_indexes}")

    def setConcatenateClasses(self, concatenate_classes: bool) -> None:
        """Sets the concatenate_classes flag for the classification sequence model.

        @param concatenate_classes: If True, concatenates consecutive classes into a
            single string. Used mostly for text processing.
        @type concatenate_classes: bool
        """
        if not isinstance(concatenate_classes, bool):
            raise ValueError("concatenate_classes must be a boolean.")
        self.concatenate_classes = concatenate_classes
        self._logger.debug(f"Concatenate classes set to {concatenate_classes}")

    def build(self, head_config: Dict[str, Any]) -> "ClassificationSequenceParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser. The required keys are `classes`, `n_classes`, and `is_softmax`.
        In addition to these, there are three optional keys that are mostly used for text processing: `ignored_indexes`, `remove_duplicates` and `concatenate_classes`.
        @type head_config: Dict[str, Any]

        @return: Returns the instantiated parser with the correct configuration.
        @rtype: ClassificationParser
        """
        super().build(head_config)
        self.ignored_indexes = head_config.get("ignored_indexes", [])
        self.remove_duplicates = head_config.get(
            "remove_duplicates", self.remove_duplicates
        )
        self.concatenate_classes = head_config.get(
            "concatenate_classes", self.concatenate_classes
        )

        self._logger.debug(
            f"ClassificationSequenceParser built with ignored_indexes={self.ignored_indexes}, remove_duplicates={self.remove_duplicates}, concatenate_classes={self.concatenate_classes}"
        )

        return self

    def run(self):
        self._logger.debug("ClassificationSequenceParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()

            except dai.MessageQueue.QueueException:
                break

            layers = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layers}")
            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            if self.n_classes == 0:
                raise ValueError("Classes must be provided for classification.")

            scores = output.getTensor(self.output_layer_name, dequantize=True).astype(
                np.float32
            )

            if len(scores.shape) != 3 and len(scores.shape) != 2:
                raise ValueError(
                    f"Scores should be a 3D or 2D array, got shape {scores.shape}."
                )

            if len(scores.shape) == 3:
                if scores.shape[0] == 1:
                    scores = scores[0]
                elif scores.shape[2] == 1:
                    scores = scores[:, :, 0]
                else:
                    raise ValueError(
                        "Scores should be a 3D array of shape (1, sequence_length, n_classes) or (sequence_length, n_classes, 1)."
                    )

            if not self.is_softmax:
                scores = softmax(scores, axis=1, keep_dims=True)

            msg = create_classification_sequence_message(
                classes=self.classes,
                scores=scores,
                remove_duplicates=self.remove_duplicates,
                ignored_indexes=self.ignored_indexes,
                concatenate_classes=self.concatenate_classes,
            )
            msg.setTimestamp(output.getTimestamp())
            msg.setSequenceNum(output.getSequenceNum())
            msg.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                msg.setTransformation(transformation)

            self._logger.debug(f"Created message with {len(msg.classes)} classes")

            self.out.send(msg)

            self._logger.debug("Classification message sent successfully")
