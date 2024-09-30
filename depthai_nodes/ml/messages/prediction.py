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
        @raise TypeError: If the predicted value is not of type float.
        """
        if not isinstance(value, float):
            raise TypeError(
                f"prediction must be of type float, instead got {type(value)}."
            )
        self._prediction = value


class Predictions(dai.Buffer):
    """Predictions class for storing predictions.

    Attributes
    ----------
    lines : List[Prediction]
        List of predictions.
    """

    def __init__(self):
        """Initializes the Predictions object."""
        super().__init__()
        self._predictions: List[Prediction] = []

    @property
    def predictions(self) -> List[Prediction]:
        """Returns the predictions.

        @return: List of predictions.
        @rtype: List[Prediction]
        """
        return self._predictions

    @predictions.setter
    def predictions(self, values: List[Prediction]):
        """Sets the predictions.

        @param value: List of predicted values.
        @type value: List[Prediction]
        @raise TypeError: If the predicted values are not a list.
        @raise TypeError: If each predicted value is not of type Prediction.
        """
        if not isinstance(values, List):
            raise TypeError(
                f"Predictions must be of type List[Prediction], instead got {type(values)}."
            )
        for value in values:
            if not isinstance(value, Prediction):
                raise TypeError(
                    f"Each prediction must be of type Prediction, instead got {type(value)}."
                )
        self._predictions = values
