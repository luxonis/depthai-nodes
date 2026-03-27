from typing import TypeVar, Union

import depthai as dai

from depthai_nodes.message.classification import Classifications
from depthai_nodes.message.clusters import Clusters
from depthai_nodes.message.lines import Lines
from depthai_nodes.message.map import Map2D
from depthai_nodes.message.prediction import Predictions
from depthai_nodes.message.segmentation import SegmentationMask

GMessage = TypeVar(
    "GMessage",
    bound=Union[
        dai.ImgDetections,
        dai.KeypointsList,
        SegmentationMask,
        Clusters,
        Map2D,
        Lines,
        Predictions,
        Classifications,
    ],
)
UNASSIGNED_MASK_LABEL = -1
