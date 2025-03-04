from depthai_nodes.utils.constants import (
    DETECTION_BORDER_THICKNESS_PER_RESOLUTION,
    KEYPOINT_THICKNESS_PER_RESOLUTION,
    TEXT_SIZE_PER_HEIGHT,
)


class AnnotationSizes:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    @property
    def border_thickness(self):
        return DETECTION_BORDER_THICKNESS_PER_RESOLUTION * (self._height + self._width)

    @property
    def keypoint_thickness(self):
        return KEYPOINT_THICKNESS_PER_RESOLUTION * (self._height + self._width)

    @property
    def text_size(self):
        return TEXT_SIZE_PER_HEIGHT * self._height

    @property
    def relative_text_size(self):
        return self.text_size / self._height

    @property
    def text_space(self):
        return self.relative_text_size / 2

    @property
    def aspect_ratio(self):
        return self._width / self._height
