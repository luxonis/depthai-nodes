from typing import TypeVar

import depthai as dai

GMessage = TypeVar(
    "GMessage",
    bound=dai.ImgDetections
    | dai.beta.Keypoints
    | dai.SegmentationMask
    | dai.beta.Clusters
    | dai.beta.Map2D
    | dai.beta.Lines
    | dai.beta.Predictions
    | dai.beta.Classifications,
)
