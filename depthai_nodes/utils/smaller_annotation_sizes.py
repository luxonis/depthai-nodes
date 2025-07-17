from depthai_nodes.constants import (
    SMALLER_DETECTION_BORDER_THICKNESS_PER_RESOLUTION,
    SMALLER_DETECTION_CORNER_SIZE,
    SMALLER_FONT_SIZE_PER_HEIGHT,
    SMALLER_KEYPOINT_THICKNESS_PER_RESOLUTION,
)

from .annotation_sizes import AnnotationSizes


class SmallerAnnotationSizes(AnnotationSizes):
    @property
    def border_thickness(self):
        return self._get_thickness(SMALLER_DETECTION_BORDER_THICKNESS_PER_RESOLUTION)

    @property
    def keypoint_thickness(self):
        return self._get_thickness(SMALLER_KEYPOINT_THICKNESS_PER_RESOLUTION)

    @property
    def font_size(self):
        return self._get_size_per_height(SMALLER_FONT_SIZE_PER_HEIGHT)

    @property
    def corner_size(self):
        return SMALLER_DETECTION_CORNER_SIZE
