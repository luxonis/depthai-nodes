import depthai as dai


def create_cluster_message(
    clusters: list[list[list[float | int]]],
) -> dai.beta.Clusters:
    """Create a native cluster message from grouped 2D points.

    Args:
        clusters: List of clusters, each a list of XY points. Points may be lists or
            tuples containing two numeric coordinates. Coordinates are copied without
            scaling; empty clusters are retained.

    Returns:
        A message with one cluster per input list, labeled by its zero-based list index.

    Raises:
        TypeError: If clusters or their point containers have unsupported types, or a
            coordinate is not an int or float.
        ValueError: If a point does not contain exactly two coordinates.
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

    message = dai.beta.Clusters()
    temp = []
    for i, cluster in enumerate(clusters):
        temp_cluster = dai.beta.Cluster()
        temp_cluster.label = i
        temp_cluster.points = dai.VectorPoint2f(
            [dai.Point2f(float(point[0]), float(point[1])) for point in cluster]
        )

        temp.append(temp_cluster)

    message.clusters = temp

    return message
