from enum import Enum
from typing import List, Tuple


class ViewportClipper:
    """Adjusts coordinates of points such that they are not outside of a specified
    viewport."""

    class _PointLocation(Enum):
        INSIDE = 0b0000
        LEFT = 0b0001
        RIGHT = 0b0010
        BOTTOM = 0b0100
        TOP = 0b1000

    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float):
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def clip_rect(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clips a rectangle defined by points to viewport.

        Uses Cohen-Sutherland line clipping algorithm for each edge of the polygon.

        @param points: List of points defining the polygon vertices
        @type points: List[Tuple[float, float]]
        @return: List of points defining the clipped polygon vertices
        @rtype: List[Tuple[float, float]]
        """
        clipped_points: List[Tuple[float, float]] = []
        points_len = len(points)
        for i in range(points_len):
            current_point = points[i]
            next_point = points[(i + 1) % points_len]
            clipped_line_pts = self.clip_line(current_point, next_point)
            if not clipped_line_pts:
                continue
            start, end = clipped_line_pts
            if start != current_point:
                clipped_points.append(start)
            else:
                clipped_points.append(current_point)
            if end != next_point:
                clipped_points.append(end)
        return clipped_points

    def clip_line(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        """Clips a line segment to viewport (0,1).

        Uses Cohen-Sutherland line clipping algorithm.

        @param pt1: Start point of the line
        @type pt1: tuple[float, float]
        @param pt2: End point of the line
        @type pt2: tuple[float, float]
        @return: Clipped line segment as (start_point, end_point) or None if line is
            completely outside of the viewport
        @rtype: tuple[tuple[float, float], tuple[float, float]] | None
        """

        # Compute codes for both points
        location1 = self._get_location(pt1)
        location2 = self._get_location(pt2)

        while True:
            if (
                location1 == self._PointLocation.INSIDE.value
                and location2 == self._PointLocation.INSIDE.value
            ):
                return pt1, pt2
            # Line completely outside viewport
            if location1 & location2 != self._PointLocation.INSIDE.value:
                return None
            # Pick an outside point
            location = location1 if location1 != 0 else location2
            # Find intersection point
            x, y = self._calculate_intersection(pt1, pt2, location)
            # Replace outside point
            if location == location1:
                pt1 = x, y
                location1 = self._get_location(pt1)
            else:
                pt2 = x, y
                location2 = self._get_location(pt2)

    def _calculate_intersection(
        self, pt1: Tuple[float, float], pt2: Tuple[float, float], location: int
    ) -> Tuple[float, float]:
        x1, y1 = pt1
        x2, y2 = pt2
        if location & self._PointLocation.TOP.value:
            x = self._interpolate(x1, x2, y1, y2, self._max_y)
            y = self._max_y
        elif location & self._PointLocation.BOTTOM.value:
            x = self._interpolate(x1, x2, y1, y2, self._min_y)
            y = self._min_y
        elif location & self._PointLocation.RIGHT.value:
            y = self._interpolate(y1, y2, x1, x2, self._max_x)
            x = self._max_x
        elif location & self._PointLocation.LEFT.value:
            y = self._interpolate(y1, y2, x1, x2, self._min_x)
            x = self._min_x
        else:
            # Should not happen if location is valid
            x, y = x1, y1
        return x, y

    def _interpolate(
        self, a1: float, a2: float, b1: float, b2: float, b_target: float
    ) -> float:
        if b2 == b1:  # No change in b dimension, return original a coordinate
            return a1
        return a1 + (a2 - a1) * (b_target - b1) / (b2 - b1)

    def _get_location(self, pt: Tuple[float, float]) -> int:
        x, y = pt
        location = self._PointLocation.INSIDE.value
        if x < self._min_x:
            location |= self._PointLocation.LEFT.value
        elif x > self._max_x:
            location |= self._PointLocation.RIGHT.value
        if y < self._min_y:
            location |= self._PointLocation.BOTTOM.value
        elif y > self._max_y:
            location |= self._PointLocation.TOP.value
        return location
