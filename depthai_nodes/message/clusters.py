import copy
from typing import List, Optional

import cv2
import depthai as dai
import numpy as np

from .utils import (
    copy_message,
)


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
        self._points: List[dai.Point2f] = []

    def copy(self):
        """Creates a new instance of the Cluster class and copies the attributes.

        @return: A new instance of the Cluster class.
        @rtype: Cluster
        """
        new_obj = Cluster()
        new_obj.label = copy.deepcopy(self.label)
        new_obj.points = [copy_message(p) for p in self.points]
        return new_obj

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
        self._transformation: Optional[dai.ImgTransformation] = None

    def copy(self):
        """Creates a new instance of the Clusters class and copies the attributes.

        @return: A new instance of the Clusters class.
        @rtype: Clusters
        """
        new_obj = Clusters()
        new_obj.clusters = [cluster.copy() for cluster in self.clusters]
        new_obj.setSequenceNum(self.getSequenceNum())
        new_obj.setTimestamp(self.getTimestamp())
        new_obj.setTimestampDevice(self.getTimestampDevice())
        new_obj.setTransformation(self.transformation)
        return new_obj

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
    def transformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self._transformation

    @transformation.setter
    def transformation(self, value: Optional[dai.ImgTransformation]):
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

    def setTransformation(self, transformation: Optional[dai.ImgTransformation]):
        """Sets the Image Transformation object.

        @param transformation: The Image Transformation object.
        @type transformation: dai.ImgTransformation
        @raise TypeError: If value is not a dai.ImgTransformation object.
        """
        self.transformation = transformation

    def getTransformation(self) -> Optional[dai.ImgTransformation]:
        """Returns the Image Transformation object.

        @return: The Image Transformation object.
        @rtype: dai.ImgTransformation
        """
        return self.transformation

    def getVisualizationMessage(self) -> dai.ImgAnnotations:
        """Creates a default visualization message for clusters and colors each one
        separately."""
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()

        num_clusters = len(self.clusters)
        color_mask = np.array(range(0, 255, 255 // num_clusters), dtype=np.uint8)
        color_mask = cv2.applyColorMap(color_mask, cv2.COLORMAP_RAINBOW)
        color_mask = color_mask / 255
        color_mask = color_mask.reshape(-1, 3)

        for i, cluster in enumerate(self.clusters):
            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.POINTS
            pointsAnnotation.points = dai.VectorPoint2f(cluster.points)
            r, g, b = color_mask[i]
            color = dai.Color(r, g, b)
            pointsAnnotation.outlineColor = color
            pointsAnnotation.fillColor = color
            pointsAnnotation.thickness = 2.0
            annotation.points.append(pointsAnnotation)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(self.getTimestamp())
        img_annotations.setSequenceNum(self.getSequenceNum())
        img_annotations.setTimestampDevice(self.getTimestampDevice())
        return img_annotations
