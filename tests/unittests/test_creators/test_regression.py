import pytest

from depthai_nodes import Prediction, Predictions
from depthai_nodes.message.creators import create_regression_message


def test_valid_input():
    predictions = [0.1, 0.2, 0.3]
    message = create_regression_message(predictions)

    assert isinstance(message, Predictions)
    assert len(message.predictions) == 3
    assert all(isinstance(pred, Prediction) for pred in message.predictions)
    assert message.predictions[0].prediction == 0.1
    assert message.predictions[1].prediction == 0.2
    assert message.predictions[2].prediction == 0.3


def test_empty_list():
    predictions = []
    message = create_regression_message(predictions)

    assert isinstance(message, Predictions)
    assert len(message.predictions) == 0


def test_invalid_type():
    with pytest.raises(ValueError):
        create_regression_message("not a list")


def test_invalid_prediction_type():
    predictions = [0.1, "not a float", 0.3]
    with pytest.raises(ValueError):
        create_regression_message(predictions)
