from typing import List, Union

import depthai_nodes.message.creators as creators

from .constants import COLLECTIONS


def create_clusters(
    clusters: List[List[List[Union[float, int]]]] = COLLECTIONS["clusters"],
):
    return creators.create_cluster_message(clusters=clusters)
