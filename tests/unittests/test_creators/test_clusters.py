import pytest

from depthai_nodes import Cluster, Clusters
from depthai_nodes.message.creators import (
    create_cluster_message,
)


def test_valid_input():
    clusters = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    message = create_cluster_message(clusters)

    assert isinstance(message, Clusters)
    assert len(message.clusters) == 2
    assert all(isinstance(cluster, Cluster) for cluster in message.clusters)
    assert message.clusters[0].label == 0
    assert message.clusters[1].label == 1
    assert len(message.clusters[0].points) == 2
    assert len(message.clusters[1].points) == 2


def test_invalid_clusters_type():
    with pytest.raises(TypeError):
        create_cluster_message("not a list")


def test_invalid_cluster_type():
    with pytest.raises(TypeError):
        create_cluster_message(["not a list"])


def test_invalid_point_type():
    with pytest.raises(TypeError):
        create_cluster_message([["not a list"]])


def test_invalid_point_length():
    with pytest.raises(ValueError):
        create_cluster_message([[[1.0, 2.0, 3.0]]])


def test_invalid_value_type():
    with pytest.raises(TypeError):
        create_cluster_message([[[1.0, "not a number"]]])


def test_clusters_input_int():
    clusters = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    create_cluster_message(clusters)


def test_clusters_input_mixed():
    clusters = [[[1, 2], [3, 4], [5, 6]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    create_cluster_message(clusters)


def test_empty_clusters():
    clusters = []
    message = create_cluster_message(clusters)

    assert isinstance(message, Clusters)
    assert len(message.clusters) == 0


def test_empty_cluster():
    clusters = [[]]
    message = create_cluster_message(clusters)

    assert isinstance(message, Clusters)
    assert len(message.clusters) == 1
    assert len(message.clusters[0].points) == 0
