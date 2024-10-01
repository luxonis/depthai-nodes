from typing import List

import depthai as dai


class Cluster(dai.Buffer):
    """Cluster class for storing a cluster.

    Attributes
    ----------
    label : int
        Label of the cluster.
    points : List[dai.Point2f]
        List of points in the cluster.
    """

    def __init__(self):
        """Initializes the Cluster object."""
        super().__init__()
        self._label: int = None
        self.points: List[dai.Point2f] = []

    @property
    def label(self) -> int:
        """Returns the label of the cluster.

        @return: Label of the cluster.
        @rtype: int
        """
        return self._label

    @label.setter
    def label(self, value: int):
        """Sets the label of the cluster.

        @param value: Label of the cluster.
        @type value: int
        @raise TypeError: If the label is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError(f"label must be of type int, instead got {type(value)}.")
        self._label = value

    @property
    def points(self) -> List[dai.Point2f]:
        """Returns the points in the cluster.

        @return: List of points in the cluster.
        @rtype: List[dai.Point2f]
        """
        return self._points

    @points.setter
    def points(self, value: List[dai.Point2f]):
        """Sets the points in the cluster.

        @param value: List of points in the cluster.
        @type value: List[dai.Point2f]
        @raise TypeError: If the points are not a list.
        @raise TypeError: If each point is not of type dai.Point2f.
        """
        if not isinstance(value, List):
            raise TypeError("points must be a list.")
        for point in value:
            if not isinstance(point, dai.Point2f):
                raise TypeError("All items in points must be of type dai.Point2f.")
        self._points = value


class Clusters(dai.Buffer):
    """Clusters class for storing clusters.

    Attributes
    ----------
    clusters : List[Cluster]
        List of clusters.
    """

    def __init__(self):
        """Initializes the Clusters object."""
        super().__init__()
        self._clusters: List[Cluster] = []

    @property
    def clusters(self) -> List[Cluster]:
        """Returns the clusters.

        @return: List of clusters.
        @rtype: List[Cluster]
        """
        return self._clusters

    @clusters.setter
    def clusters(self, value: List[Cluster]):
        """Sets the clusters.

        @param value: List of clusters.
        @type value: List[Cluster]
        @raise TypeError: If the clusters are not a list.
        @raise TypeError: If each cluster is not of type Cluster.
        """
        if not isinstance(value, List):
            raise TypeError("clusters must be a list.")
        for cluster in value:
            if not isinstance(cluster, Cluster):
                raise TypeError("All items in clusters must be of type Cluster.")
        self._clusters = value
