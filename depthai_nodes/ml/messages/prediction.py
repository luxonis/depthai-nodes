from typing import List

import depthai as dai


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
        self._transformation: dai.ImgTransformation = None

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
    def transformation(self) -> dai.ImgTransformation:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: dai.ImgTransformation):
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
