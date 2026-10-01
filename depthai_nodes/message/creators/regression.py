import depthai as dai


def create_regression_message(predictions: list[float]) -> dai.beta.Predictions:
    """Create a DepthAI message for prediction models.

    @param predictions: Predicted value(s).
    @type predictions: list[float]
    @return: Predictions message containing the predicted value(s).
    @rtype: dai.beta.Predictions
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
        prediction_object = dai.beta.Prediction()
        prediction_object.prediction = prediction
        prediction_objects_list.append(prediction_object)

    predictions_message = dai.beta.Predictions()
    predictions_message.predictions = prediction_objects_list

    return predictions_message
