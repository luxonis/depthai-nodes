from depthai_nodes.utils.viewport_clipper import ViewportClipper


def test_clip_line_both_inside():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (0.2, 0.3)
    pt2 = (0.7, 0.8)

    result = clipper.clip_line(pt1, pt2)
    assert result == (pt1, pt2)


def test_clip_line_both_outside_visible():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (-0.5, 0.5)
    pt2 = (1.5, 0.5)

    result = clipper.clip_line(pt1, pt2)
    assert result is not None
    start, end = result
    assert start == (0.0, 0.5)  # Left boundary intersection
    assert end == (1.0, 0.5)  # Right boundary intersection


def test_clip_line_both_outside_invisible():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (-0.5, -0.5)
    pt2 = (-0.3, -0.3)

    result = clipper.clip_line(pt1, pt2)
    assert result is None  # Line completely outside


def test_clip_line_one_inside_one_outside():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (0.5, 0.5)
    pt2 = (1.5, 1.5)

    result = clipper.clip_line(pt1, pt2)
    assert result is not None
    start, end = result
    assert start == (0.5, 0.5)  # Original inside point
    assert end == (1.0, 1.0)  # Boundary intersection


def test_clip_line_horizontal():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (-0.5, 0.5)
    pt2 = (1.5, 0.5)

    result = clipper.clip_line(pt1, pt2)
    assert result is not None
    start, end = result
    assert start == (0.0, 0.5)
    assert end == (1.0, 0.5)


def test_clip_line_vertical():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (0.5, -0.5)
    pt2 = (0.5, 1.5)

    result = clipper.clip_line(pt1, pt2)
    assert result is not None
    start, end = result
    assert start == (0.5, 0.0)
    assert end == (0.5, 1.0)


def test_clip_line_diagonal_intersection():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    pt1 = (0.5, 0.5)
    pt2 = (1.5, 1.5)

    result = clipper.clip_line(pt1, pt2)
    assert result is not None
    start, end = result
    assert start == (0.5, 0.5)  # Original inside point
    assert end == (1.0, 1.0)  # Boundary intersection


def test_clip_line_edge_cases():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    # Line exactly on boundary
    result = clipper.clip_line((0.0, 0.5), (1.0, 0.5))
    assert result == ((0.0, 0.5), (1.0, 0.5))

    # Line with one point exactly on corner
    result = clipper.clip_line((0.0, 0.0), (0.5, 0.5))
    assert result == ((0.0, 0.0), (0.5, 0.5))

    # Very long diagonal line
    result = clipper.clip_line((-1000.0, -1000.0), (1000.0, 1000.0))
    assert result is not None
    start, end = result
    assert start == (0.0, 0.0)
    assert end == (1.0, 1.0)


def test_clip_line_division_by_zero_protection():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    # Horizontal line (dy = 0)
    result = clipper.clip_line((-0.5, 0.5), (1.5, 0.5))
    assert result is not None

    # Vertical line (dx = 0)
    result = clipper.clip_line((0.5, -0.5), (0.5, 1.5))
    assert result is not None

    # Degenerate line (same point)
    result = clipper.clip_line((0.5, 0.5), (0.5, 0.5))
    assert result == ((0.5, 0.5), (0.5, 0.5))


def test_clip_rect_simple_inside():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    result = clipper.clip_rect(points)

    assert result == points


def test_clip_rect_simple_outside():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(1.5, 1.5), (2.0, 1.5), (2.0, 2.0), (1.5, 2.0)]
    result = clipper.clip_rect(points)

    assert len(result) == 0


def test_clip_rect_partially_overlapping():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    result = clipper.clip_rect(points)

    expected = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_surrounding_viewport():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)]
    result = clipper.clip_rect(points)

    # Should return the viewport rectangle when input completely surrounds it
    expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_empty_input():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)
    result = clipper.clip_rect([])
    assert result == []


def test_clip_rect_with_different_viewport():
    clipper = ViewportClipper(-2.0, 2.0, -2.0, 2.0)

    # Rectangle inside this viewport
    points = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    result = clipper.clip_rect(points)

    assert len(result) == 4
    assert set(result) == set(points)


def test_clip_rect_extends_all_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_right_boundary():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.7, 0.3), (1.3, 0.3), (1.3, 0.7), (0.7, 0.7)]
    result = clipper.clip_rect(points)

    expected = [(0.7, 0.3), (1.0, 0.3), (1.0, 0.7), (0.7, 0.7)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_left_boundary():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, 0.3), (0.3, 0.3), (0.3, 0.7), (-0.3, 0.7)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.3), (0.3, 0.3), (0.3, 0.7), (0.0, 0.7)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_bottom_boundary():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.3, -0.3), (0.7, -0.3), (0.7, 0.3), (0.3, 0.3)]
    result = clipper.clip_rect(points)

    expected = [(0.3, 0.0), (0.7, 0.0), (0.7, 0.3), (0.3, 0.3)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_top_boundary():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.3, 0.7), (0.7, 0.7), (0.7, 1.3), (0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.3, 0.7), (0.7, 0.7), (0.7, 1.0), (0.3, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_left_right_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, 0.3), (1.3, 0.3), (1.3, 0.7), (-0.3, 0.7)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.3), (1.0, 0.3), (1.0, 0.7), (0.0, 0.7)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_bottom_top_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.3, -0.3), (0.7, -0.3), (0.7, 1.3), (0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_left_bottom_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, -0.3), (0.5, -0.3), (0.5, 0.5), (-0.3, 0.5)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_right_bottom_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.5, -0.3), (1.3, -0.3), (1.3, 0.5), (0.5, 0.5)]
    result = clipper.clip_rect(points)

    expected = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_left_top_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, 0.5), (0.5, 0.5), (0.5, 1.3), (-0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_right_top_boundaries():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.5, 0.5), (1.3, 0.5), (1.3, 1.3), (0.5, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_three_boundaries_except_right():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, -0.3), (0.5, -0.3), (0.5, 1.3), (-0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_three_boundaries_except_left():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.5, -0.3), (1.3, -0.3), (1.3, 1.3), (0.5, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_three_boundaries_except_bottom():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, 0.5), (1.3, 0.5), (1.3, 1.3), (-0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_extends_three_boundaries_except_top():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, -0.3), (1.3, -0.3), (1.3, 0.5), (-0.3, 0.5)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_single_point_inside():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    result = clipper.clip_rect(points)

    expected = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_diagonal_crossing():
    """Test clipping a rectangle that crosses viewport diagonally."""
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    # Rectangle that crosses diagonally from bottom-left outside to top-right outside
    points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    result = clipper.clip_rect(points)

    # Rectangle extends beyond left and bottom boundaries
    expected = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_diagonal_crossing_opposite():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(-0.3, 0.7), (0.3, 0.7), (0.3, 1.3), (-0.3, 1.3)]
    result = clipper.clip_rect(points)

    expected = [(0.0, 0.7), (0.3, 0.7), (0.3, 1.0), (0.0, 1.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_with_negative_viewport():
    clipper = ViewportClipper(min_x=-1.0, max_x=0.0, min_y=-1.0, max_y=0.0)

    points = [(-1.5, -1.5), (0.5, -1.5), (0.5, 0.5), (-1.5, 0.5)]
    result = clipper.clip_rect(points)

    expected = [(-1.0, -1.0), (0.0, -1.0), (0.0, 0.0), (-1.0, 0.0)]
    assert len(result) == 4
    assert set(result) == set(expected)


def test_clip_rect_very_small_rectangle():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.45, 0.45), (0.55, 0.45), (0.55, 0.55), (0.45, 0.55)]
    result = clipper.clip_rect(points)

    assert len(result) == 4
    assert set(result) == set(points)


def test_clip_rect_touching_boundary():
    clipper = ViewportClipper(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    result = clipper.clip_rect(points)

    assert len(result) == 4
    assert set(result) == set(points)
