import pytest
import numpy as np

from depthai_nodes.ml.messages.creators.regression import create_regression_message
from depthai_nodes.ml.messages import Predictions

np.random.seed(0)


def test_not_list_predictions():
    predictions = 10
    with pytest.raises(ValueError, match=f"Predictions should be list, got <class 'int'>."):
        create_regression_message(predictions)


def test_not_float_predictions():
    predictions = np.random.randint(0,10,5).tolist()
    with pytest.raises(ValueError, match=f"Each prediction should be a float, got <class 'int'> instead."):
        create_regression_message(predictions)


def test_predictions_object():
    predictions = np.random.rand(10).tolist()

    message = create_regression_message(predictions)

    assert isinstance(message, Predictions)
    assert predictions[0] == message.predictions[0].prediction


if __name__ == "__main__":
    pytest.main()
