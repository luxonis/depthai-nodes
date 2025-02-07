from typing import List

from depthai_nodes import Prediction, Predictions


def create_regression_message(predictions: List[float]) -> Predictions:
    """Create a DepthAI message for prediction models.

    @param predictions: Predicted value(s).
    @type predictions: List[float]
    @return: Predictions message containing the predicted value(s).
    @rtype: Predictions
    @raise ValueError: If predictions is not a list.
    @raise ValueError: If each prediction is not a float.
    """

    if not isinstance(predictions, list):
        raise ValueError(f"Predictions should be list, got {type(predictions)}.")

    for prediction in predictions:
        if not isinstance(prediction, float):
            raise ValueError(
                f"Each prediction should be a float, got {type(prediction)} instead."
            )

    prediction_objects_list = []
    for prediction in predictions:
        prediction_object = Prediction()
        prediction_object.prediction = prediction
        prediction_objects_list.append((prediction_object))

    predictions_message = Predictions()
    predictions_message.predictions = prediction_objects_list

    return predictions_message
