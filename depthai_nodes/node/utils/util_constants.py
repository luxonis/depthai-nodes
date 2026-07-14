from typing import TypeVar

import depthai as dai

from depthai_nodes.message.classification import Classifications
from depthai_nodes.message.clusters import Clusters
from depthai_nodes.message.keypoints import Keypoints
from depthai_nodes.message.lines import Lines
from depthai_nodes.message.map import Map2D
from depthai_nodes.message.prediction import Predictions

GMessage = TypeVar(
    "GMessage",
    bound=dai.ImgDetections
    | Keypoints
    | dai.SegmentationMask
    | Clusters
    | Map2D
    | Lines
    | Predictions
    | Classifications,
)
UNASSIGNED_MASK_LABEL = -1
