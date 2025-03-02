from enum import Enum
from typing import List, Tuple


class ViewportClipper:
    """Adjusts coordinates of points so they are not outside of the viewport (0,1)."""

    class _PointLocation(Enum):
        INSIDE = 0b0000
        LEFT = 0b0001
        RIGHT = 0b0010
        BOTTOM = 0b0100
        TOP = 0b1000

    @staticmethod
    def clip_rect(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        clipped_points: List[Tuple[float, float]] = []
        points_len = len(points)
        for i in range(points_len):
            current_point = points[i]
            next_point = points[(i + 1) % points_len]
            clipped_line_pts = ViewportClipper.clip_line(current_point, next_point)
            if not clipped_line_pts:
                continue
            start, end = clipped_line_pts
            if start != current_point:
                clipped_points.append(start)
            else:
                clipped_points.append(current_point)
            if end != next_point:
                clipped_points.append(end)
            else:
                clipped_points.append(next_point)
        return clipped_points

    @staticmethod
    def clip_line(pt1: Tuple[float, float], pt2: Tuple[float, float]):
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
        x1, y1 = pt1
        x2, y2 = pt2
        code1 = ViewportClipper._get_location(x1, y1)
        code2 = ViewportClipper._get_location(x2, y2)

        while True:
            # Both points inside viewport
            if code1 == 0 and code2 == 0:
                return (x1, y1), (x2, y2)

            # Line completely outside viewport
            if code1 & code2 != 0:
                return None

            # Pick an outside point
            code = code1 if code1 != 0 else code2

            # Find intersection point
            if code & ViewportClipper._PointLocation.TOP.value:
                x = x1 + (x2 - x1) * (1 - y1) / (y2 - y1)
                y = 1.0
            elif code & ViewportClipper._PointLocation.BOTTOM.value:
                x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                y = 0.0
            elif code & ViewportClipper._PointLocation.RIGHT.value:
                y = y1 + (y2 - y1) * (1 - x1) / (x2 - x1)
                x = 1.0
            elif code & ViewportClipper._PointLocation.LEFT.value:
                y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                x = 0.0

            # Replace outside point
            if code == code1:
                x1, y1 = x, y
                code1 = ViewportClipper._get_location(x1, y1)
            else:
                x2, y2 = x, y
                code2 = ViewportClipper._get_location(x2, y2)

    @staticmethod
    def _get_location(x: float, y: float) -> int:
        location = ViewportClipper._PointLocation.INSIDE.value
        if x < 0:
            location |= ViewportClipper._PointLocation.LEFT.value
        elif x > 1:
            location |= ViewportClipper._PointLocation.RIGHT.value
        if y < 0:
            location |= ViewportClipper._PointLocation.BOTTOM.value
        elif y > 1:
            location |= ViewportClipper._PointLocation.TOP.value
        return location
