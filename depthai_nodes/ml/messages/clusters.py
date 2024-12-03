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
        @raise TypeError: If value is not an int.
        """
        if not isinstance(value, int):
            raise TypeError(f"Label must be of type int, instead got {type(value)}.")
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
        @raise TypeError: If value is not a list.
        @raise TypeError: If each element is not of type dai.Point2f.
        """
        if not isinstance(value, List):
            raise TypeError(f"Points must be a list, instead got {type(value)}.")
        if not all(isinstance(point, dai.Point2f) for point in value):
            raise ValueError("Points must be a list of dai.Point2f objects")
        self._points = value


class Clusters(dai.Buffer):
    """Clusters class for storing clusters.

    Attributes
    ----------
    clusters : List[Cluster]
        List of clusters.
    transformation : dai.ImgTransformation
        Image transformation object.
    """

    def __init__(self):
        """Initializes the Clusters object."""
        super().__init__()
        self._clusters: List[Cluster] = []
        self._transformation: dai.ImgTransformation = None

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
        @raise TypeError: If value is not a list.
        @raise ValueError: If each element is not of type Cluster.
        """
        if not isinstance(value, List):
            raise TypeError("Clusters must be a list.")
        if not all(isinstance(cluster, Cluster) for cluster in value):
            raise ValueError("Clusters must be a list of Cluster objects.")
        self._clusters = value

    @property
    def transformation(self) -> dai.ImgTransformation:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: dai.ImgTransformation):
        """Sets the Image Transformation object.

        @param value: The Image Transformation object.
        @type value: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """

        if value is not None:
            if not isinstance(value, dai.ImgTransformation):
                raise TypeError(
                    f"Transformation must be a dai.ImgTransformation object, instead got {type(value)}."
                )
        self._transformation = value
