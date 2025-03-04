import depthai as dai
import numpy as np
import pytest

from depthai_nodes import Classifications


@pytest.fixture
def classifications():
    return Classifications()


def test_initialization(classifications: Classifications):
    assert classifications.classes == []
    assert np.array_equal(classifications.scores, np.array([]))
    assert classifications.transformation is None


def test_set_classes(classifications: Classifications):
    classes = ["cat", "dog", "bird"]
    classifications.classes = classes
    assert classifications.classes == classes

    with pytest.raises(TypeError):
        classifications.classes = "not a list"

    with pytest.raises(ValueError):
        classifications.classes = ["cat", 123, "bird"]


def test_set_scores(classifications: Classifications):
    scores = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    classifications.scores = scores
    assert np.array_equal(classifications.scores, scores)

    with pytest.raises(TypeError):
        classifications.scores = "not an array"

    with pytest.raises(ValueError):
        classifications.scores = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)

    with pytest.raises(ValueError):
        classifications.scores = np.array([0.9, 0.05, "not a float"], dtype=object)


def test_top_class(classifications: Classifications):
    classes = ["cat", "dog", "bird"]
    classifications.classes = classes
    assert classifications.top_class == "cat"


def test_top_score(classifications: Classifications):
    scores = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    classifications.scores = scores
    assert np.allclose(classifications.top_score, 0.9, atol=1e-3)


def test_set_transformation(classifications: Classifications):
    transformation = dai.ImgTransformation()
    classifications.transformation = transformation
    assert classifications.transformation == transformation

    with pytest.raises(TypeError):
        classifications.transformation = "not a dai.ImgTransformation"


def test_set_transformation_none(classifications: Classifications):
    classifications.transformation = None
    assert classifications.transformation is None
