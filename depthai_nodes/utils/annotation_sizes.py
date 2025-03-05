from depthai_nodes.utils.constants import (
    DETECTION_BORDER_THICKNESS_PER_RESOLUTION,
    DETECTION_CORNER_SIZE,
    KEYPOINT_THICKNESS_PER_RESOLUTION,
    TEXT_SIZE_PER_HEIGHT,
)


class AnnotationSizes:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    @property
    def border_thickness(self):
        return self._get_thickness(DETECTION_BORDER_THICKNESS_PER_RESOLUTION)

    @property
    def keypoint_thickness(self):
        return self._get_thickness(KEYPOINT_THICKNESS_PER_RESOLUTION)

    def _get_thickness(self, thickness_per_resolution: float):
        return thickness_per_resolution * (self._height + self._width)

    @property
    def text_size(self):
        return self._get_size_per_height(TEXT_SIZE_PER_HEIGHT)

    def _get_size_per_height(self, size_per_heigh):
        return size_per_heigh * self._height

    @property
    def relative_text_size(self):
        return self.text_size / self._height

    @property
    def text_space(self):
        return self.relative_text_size / 2

    @property
    def aspect_ratio(self):
        return self._width / self._height

    @property
    def corner_size(self):
        return DETECTION_CORNER_SIZE
