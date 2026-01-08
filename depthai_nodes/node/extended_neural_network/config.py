from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import depthai as dai
import numpy as np


@dataclass(kw_only=True)
class TilingConfig:
    input_size: Tuple[int, int]
    grid_size: Tuple[int, int] = (2, 2)
    overlap: float = 0.1
    global_detection: bool = False
    grid_matrix: Optional[np.ndarray] = None
    iou_threshold: float = 0.2


@dataclass(kw_only=True)
class DetectionFilterConfig:
    confidence_threshold: Optional[float]
    labels_to_keep: Optional[Sequence[int]]
    labels_to_reject: Optional[Sequence[int]]
    max_detections: Optional[int]
