import pytest

from depthai_nodes.ml.messages import Classifications, MiscellaneousMessage
from depthai_nodes.ml.messages.creators.misc import create_multi_classification_message


def test_incorect_lengths():
    with pytest.raises(
        ValueError,
        match="Number of classification attributes, scores and labels should be equal. Got 1 attributes, 2 scores and 2 labels.",
    ):
        create_multi_classification_message(
            ["vehicle_type"],
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [["car", "truck"], ["red", "blue"]],
        )


def test_incorect_score_label_lengths():
    with pytest.raises(
        ValueError,
        match="Number of scores and labels should be equal for each classification attribute, got 4 scores, 2 labels for attribute vehicle_type.",
    ):
        create_multi_classification_message(
            ["vehicle_type", "vehicle_color"],
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6]],
            [["car", "truck"], ["red", "blue"]],
        )


def test_correct_usage():
    attrs = ["vehicle_type", "vehicle_color"]
    scores = [[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.4, 0.2, 0.2, 0.1]]
    names = [
        ["car", "truck", "van", "bike"],
        ["red", "blue", "green", "black", "white", "yellow"],
    ]

    res = create_multi_classification_message(attrs, scores, names)

    assert isinstance(res, MiscellaneousMessage)
    res = res.getData()
    assert isinstance(res["vehicle_type"], Classifications)
    assert isinstance(res["vehicle_color"], Classifications)
    assert res["vehicle_type"].classes == ["bike", "van", "truck", "car"]
    assert res["vehicle_type"].scores == [0.4, 0.3, 0.2, 0.1]
    assert res["vehicle_color"].classes == [
        "green",
        "black",
        "white",
        "blue",
        "yellow",
        "red",
    ]
    assert res["vehicle_color"].scores == [0.4, 0.2, 0.2, 0.1, 0.1, 0.0]


if __name__ == "__main__":
    pytest.main()
