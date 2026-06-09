import depthai_nodes.message.creators as creators

from .constants import COLLECTIONS


def create_clusters(
    clusters: list[list[list[float | int]]] = COLLECTIONS["clusters"],
):
    return creators.create_cluster_message(clusters=clusters)
