import pytest

from depthai_nodes.ml.messages.creators.clusters import create_cluster_message


def test_non_list_input():
    with pytest.raises(TypeError):
        create_cluster_message(1)


def test_non_list_of_clusters_input():
    with pytest.raises(TypeError):
        create_cluster_message([1, 2, 3])


def test_clusters_input_int():
    clusters = [[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10), (11, 12)]]
    create_cluster_message(clusters)


def test_clusters_input_float():
    clusters = [
        [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        [(7.0, 8.0), (9.0, 10.0), (11.0, 12.0)],
    ]
    create_cluster_message(clusters)


def test_clusters_input_mixed():
    clusters = [[(1, 2), (3, 4), (5, 6)], [(7.0, 8.0), (9.0, 10.0), (11.0, 12.0)]]
    create_cluster_message(clusters)


def test_clusters_input_empty():
    clusters = []
    create_cluster_message(clusters)


def test_clusters_input_empty_cluster():
    clusters = [[]]
    create_cluster_message(clusters)


def test_clusters_input_empty_point():
    with pytest.raises(TypeError):
        clusters = [[[]]]
        create_cluster_message(clusters)


def test_clusters_input_empty_point_value():
    with pytest.raises(TypeError):
        clusters = [[[1, 2], []]]
        create_cluster_message(clusters)


def test_clusters_input_point_shape():
    with pytest.raises(ValueError):
        clusters = [
            [(1, 2, 3), (3, 4, 5), (5, 6, 7)],
            [(7, 8, 9), (9, 10, 11), (11, 12, 13)],
        ]
        create_cluster_message(clusters)
