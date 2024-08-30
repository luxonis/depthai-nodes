import numpy as np
import pytest

from depthai_nodes.ml.messages import AgeGender
from depthai_nodes.ml.messages.creators.misc import create_age_gender_message


def test_wrong_age():
    with pytest.raises(ValueError, match="Age should be float, got <class 'int'>."):
        create_age_gender_message(1, [0.5, 0.5])


def test_gender_wrong():
    with pytest.raises(
        ValueError, match="Gender_prob should be list, got <class 'str'>."
    ):
        create_age_gender_message(1.0, "female")


def test_gender_empty():
    with pytest.raises(
        ValueError, match="Gender_prob list should have 2 values, got 0"
    ):
        create_age_gender_message(1.0, [])


def test_gender_types():
    with pytest.raises(
        ValueError,
        match="Gender_prob list values must be of type float, instead got <class 'str'>.",
    ):
        create_age_gender_message(1.0, [0.5, "0.5"])


def test_is_probability():
    with pytest.raises(
        ValueError,
        match="Gender_prob list must contain probabilities and sum to 1, got sum 3.7.",
    ):
        create_age_gender_message(1.0, [1.2, 2.5])


def test_correct_types():
    age = 32.4
    gender = [0.35, 0.65]
    message = create_age_gender_message(age, gender)

    assert isinstance(message, AgeGender)
    assert message.age == age
    assert message.gender.classes == ["female", "male"]
    assert np.all(np.isclose(message.gender.scores, gender))


if __name__ == "__main__":
    pytest.main()
