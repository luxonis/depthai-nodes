from depthai_nodes.constants import (
    DETECTION_BORDER_THICKNESS_PER_RESOLUTION,
    DETECTION_CORNER_SIZE,
    FONT_SIZE_PER_HEIGHT,
    KEYPOINT_THICKNESS_PER_RESOLUTION,
)


class AnnotationSizes:
    """Compute annotation dimensions from an image resolution.

    Pixel sizes scale with image height or the sum of width and height. Relative font
    sizes and spacing use image height as their reference.
    """

    def __init__(self, width: int, height: int):
        """Set the image resolution used by size calculations.

        Args:
            width: Image width in pixels.
            height: Image height in pixels; must be nonzero for relative size
                calculations.
        """
        self._width = width
        self._height = height

    @property
    def border_thickness(self):
        """Detection border thickness in pixels."""
        return self._get_thickness(DETECTION_BORDER_THICKNESS_PER_RESOLUTION)

    @property
    def keypoint_thickness(self):
        """Keypoint marker thickness in pixels."""
        return self._get_thickness(KEYPOINT_THICKNESS_PER_RESOLUTION)

    def _get_thickness(self, thickness_per_resolution: float):
        return thickness_per_resolution * (self._height + self._width)

    @property
    def font_size(self):
        """Font size in pixels, proportional to image height."""
        return self._get_size_per_height(FONT_SIZE_PER_HEIGHT)

    def _get_size_per_height(self, size_per_height):
        return size_per_height * self._height

    @property
    def relative_font_size(self):
        """Font size divided by image height."""
        return self.font_size / self._height

    @property
    def font_space(self):
        """Text spacing equal to half the relative font size."""
        return self.relative_font_size / 2

    @property
    def aspect_ratio(self):
        """Image width divided by image height."""
        return self._width / self._height

    @property
    def corner_size(self):
        """Configured relative size of detection corners."""
        return DETECTION_CORNER_SIZE
