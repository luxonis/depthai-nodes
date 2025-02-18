import depthai as dai
import pytest

from depthai_nodes import Prediction, Predictions


@pytest.fixture
def prediction():
    return Prediction()


@pytest.fixture
def predictions():
    return Predictions()


def test_prediction_initialization(prediction: Prediction):
    assert prediction.prediction is None


def test_prediction_set_prediction(prediction: Prediction):
    prediction.prediction = 0.9
    assert prediction.prediction == 0.9

    with pytest.raises(TypeError):
        prediction.prediction = "not a float"


def test_predictions_initialization(predictions: Predictions):
    assert predictions.predictions == []
    assert predictions.transformation is None


def test_predictions_set_predictions(predictions: Predictions):
    pred1 = Prediction()
    pred1.prediction = 0.1
    pred2 = Prediction()
    pred2.prediction = 0.2
    predictions_list = [pred1, pred2]
    predictions.predictions = predictions_list
    assert predictions.predictions == predictions_list

    with pytest.raises(TypeError):
        predictions.predictions = "not a list"

    with pytest.raises(ValueError):
        predictions.predictions = [pred1, "not a Prediction"]


def test_predictions_get_prediction(predictions: Predictions):
    pred1 = Prediction()
    pred1.prediction = 0.1
    predictions.predictions = [pred1]
    assert predictions.prediction == 0.1


def test_predictions_set_transformation(predictions: Predictions):
    transformation = dai.ImgTransformation()
    predictions.transformation = transformation
    assert predictions.transformation == transformation

    with pytest.raises(TypeError):
        predictions.transformation = "not a dai.ImgTransformation"


def test_predictions_set_transformation_none(predictions: Predictions):
    predictions.transformation = None
    assert predictions.transformation is None
