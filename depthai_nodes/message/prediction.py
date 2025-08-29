import copy
from typing import List, Optional

import depthai as dai

from depthai_nodes import FONT_BACKGROUND_COLOR, FONT_COLOR
from depthai_nodes.utils import AnnotationHelper, AnnotationSizes


class Prediction(dai.Buffer):
    """Prediction class for storing a prediction.

    Attributes
    ----------
    prediction : float
        The predicted value.
    """

    def __init__(self):
        """Initializes the Prediction object."""
        super().__init__()
        self._prediction: float = None

    def copy(self):
        """Creates a new instance of the Prediction class and copies the attributes.

        @return: A new instance of the Prediction class.
        @rtype: Prediction
        """
        new_obj = Prediction()
        new_obj.prediction = copy.deepcopy(self.prediction)
        return new_obj

    @property
    def prediction(self) -> float:
        """Returns the prediction.

        @return: The predicted value.
        @rtype: float
        """
        return self._prediction

    @prediction.setter
    def prediction(self, value: float):
        """Sets the prediction.

        @param value: The predicted value.
        @type value: float
        @raise TypeError: If value is not of type float.
        """
        if not isinstance(value, float):
            raise TypeError(
                f"Prediction must be of type float, instead got {type(value)}."
            )
        self._prediction = value


class Predictions(dai.Buffer):
    """Predictions class for storing predictions.

    Attributes
    ----------
    predictions : List[Prediction]
        List of predictions.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Predictions object."""
        super().__init__()
        self._predictions: List[Prediction] = []
        self._transformation: Optional[dai.ImgTransformation] = None

    def copy(self):
        """Creates a new instance of the Predictions class and copies the attributes.

        @return: A new instance of the Predictions class.
        @rtype: Predictions
        """
        new_obj = Predictions()
        new_obj.predictions = [prediction.copy() for prediction in self.predictions]
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.transformation)
        return new_obj

    @property
    def predictions(self) -> List[Prediction]:
        """Returns the predictions.

        @return: List of predictions.
        @rtype: List[Prediction]
        """
        return self._predictions

    @predictions.setter
    def predictions(self, value: List[Prediction]):
        """Sets the predictions.

        @param value: List of predicted values.
        @type value: List[Prediction]
        @raise TypeError: If value is not a list.
        @raise ValueError: If each element is not of type Prediction.
        """
        if not isinstance(value, List):
            raise TypeError(
                f"Predictions must be of type list, instead got {type(value)}."
            )
        if not all(isinstance(item, Prediction) for item in value):
            raise ValueError("Predictions must be a list of Prediction objects.")
        self._predictions = value

    @property
    def prediction(self) -> float:
        """Returns the first prediction. Useful for single predictions.

        @return: The predicted value.
        @rtype: float
        """
        return self._predictions[0].prediction

    @property
    def transformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: Optional[dai.ImgTransformation]):
        """Sets the Image Transformation object.

        @param value: The Image Transformation object.
        @type value: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """

        if value is not None:
            if not isinstance(value, dai.ImgTransformation):
                raise TypeError(
                    f"Transformation must be a dai.ImgTransformation object, instead got {type(value)}."
                )
        self._transformation = value

    def setTransformation(self, transformation: Optional[dai.ImgTransformation]):
        """Sets the Image Transformation object.

        @param transformation: The Image Transformation object.
        @type transformation: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """
        self.transformation = transformation

    def getTransformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self.transformation

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Returns the visualization message for the predictions.

        The message adds text representing the predictions to the right of the image.
        """
        if self.transformation is None:
            raise ValueError("Transformation must be set to get visualization message.")
        w, h = self.transformation.getSize()
        annotation_helper = AnnotationHelper()
        annotation_sizes = AnnotationSizes(w, h)

        x_offset = 3 / w
        y_offset = 3 / h

        for i, prediction in enumerate(self.predictions):
            y_position = (
                y_offset
                + (annotation_sizes.relative_font_size)
                + i * (annotation_sizes.relative_font_size)
            )
            annotation_helper.draw_text(
                text=f"{prediction.prediction:.2f}",
                position=(x_offset, y_position),
                color=FONT_COLOR,
                background_color=FONT_BACKGROUND_COLOR,
                size=annotation_sizes.font_size,
            )
        return annotation_helper.build(
            timestamp=self.getTimestamp(),
            sequence_num=self.getSequenceNum(),
        )
