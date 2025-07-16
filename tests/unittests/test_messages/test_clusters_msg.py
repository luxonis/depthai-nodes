import depthai as dai
import pytest

from depthai_nodes import Cluster, Clusters


@pytest.fixture
def cluster():
    return Cluster()


@pytest.fixture
def clusters():
    return Clusters()


def test_cluster_initialization(cluster: Cluster):
    assert cluster.label is None
    assert cluster.points == []


def test_cluster_set_label(cluster: Cluster):
    cluster.label = 1
    assert cluster.label == 1

    with pytest.raises(TypeError):
        cluster.label = "not an int"


def test_cluster_set_points(cluster: Cluster):
    points = [dai.Point2f(0.1, 0.2), dai.Point2f(0.3, 0.4)]
    cluster.points = points
    assert cluster.points == points

    with pytest.raises(TypeError):
        cluster.points = "not a list"

    with pytest.raises(ValueError):
        cluster.points = [dai.Point2f(0.1, 0.2), "not a Point2f"]


def test_clusters_initialization(clusters: Clusters):
    assert clusters.clusters == []
    assert clusters.transformation is None


def test_clusters_set_clusters(clusters: Clusters):
    cluster1 = Cluster()
    cluster2 = Cluster()
    clusters_list = [cluster1, cluster2]
    clusters.clusters = clusters_list
    assert clusters.clusters == clusters_list

    with pytest.raises(TypeError):
        clusters.clusters = "not a list"

    with pytest.raises(ValueError):
        clusters.clusters = [cluster1, "not a Cluster"]


def test_clusters_set_transformation(clusters: Clusters):
    transformation = dai.ImgTransformation()
    clusters.transformation = transformation
    assert clusters.transformation == transformation

    with pytest.raises(TypeError):
        clusters.transformation = "not a dai.ImgTransformation"


def test_clusters_set_transformation_none(clusters: Clusters):
    clusters.transformation = None
    assert clusters.transformation is None
