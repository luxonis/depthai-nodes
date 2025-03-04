from typing import List, Union

import depthai as dai

from depthai_nodes import Cluster, Clusters


def create_cluster_message(clusters: List[List[List[Union[float, int]]]]) -> Clusters:
    """Create a DepthAI message for clusters.

    @param clusters: List of clusters. Each cluster is a list of points with x and y
        coordinates.
    @type clusters: List[List[List[Union[float, int]]]]
    @return: Clusters message containing the detected clusters.
    @rtype: Clusters
    @raise TypeError: If the clusters are not a list.
    @raise TypeError: If each cluster is not a list.
    @raise TypeError: If each point is not a list.
    @raise TypeError: If each value in the point is not an int or float.
    """

    if not isinstance(clusters, list):
        raise TypeError(f"clusters must be a list, got {type(clusters)}")
    for cluster in clusters:
        if not isinstance(cluster, list):
            raise TypeError(f"All clusters must be of type List, got {type(cluster)}")
        for point in cluster:
            if not isinstance(point, tuple) and not isinstance(point, list):
                raise TypeError(
                    f"All points in clusters must be of type tuple or list, got {type(point)}"
                )
            if len(point) != 2:
                raise ValueError(f"Each point must have 2 values, got {len(point)}")
            for value in point:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        f"All items in points must be of type int or float, got {type(value)}"
                    )

    message = Clusters()
    temp = []
    for i, cluster in enumerate(clusters):
        temp_cluster = Cluster()
        temp_cluster.label = i
        temp_cluster.points = [
            dai.Point2f(float(point[0]), float(point[1])) for point in cluster
        ]

        temp.append(temp_cluster)

    message.clusters = temp

    return message
