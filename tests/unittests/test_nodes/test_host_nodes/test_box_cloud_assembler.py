from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import BoxCloud, BoxClouds, BoxViewDetection
from depthai_nodes.node.box_associator import box_cloud_assembler
from depthai_nodes.node.box_associator.box_cloud_assembler import (
    LINK_DISTANCE_MM,
    MAX_ABSENT_FRAMES,
    MERGE_VOXEL_SIZE_MM,
    MIN_LINK_FRACTION,
    BoxCloudAssembler,
    _choose_group,
    _Claim,
)
from depthai_nodes.node.box_dimensioner.box_geometry import BoxGeometry
from tests.utils import OutputMock
from tests.utils.box_scenes import (
    Cuboid,
    box_detection,
    box_detections,
    box_group,
    different_track_ids,
    label_cloud,
    opposite_sides,
    paint_view,
    partial_view,
    sample_faces,
    two_separate_boxes,
    two_touching_boxes,
)

pytest.importorskip("open3d")


def run(groups: list[dai.MessageGroup]) -> list[BoxClouds]:
    """Feed one group per frame and return the ``BoxClouds`` sent for each."""
    group_out = OutputMock()
    assembler = BoxCloudAssembler().build(group_out)
    assert assembler._linked_args == (group_out,)
    queue = assembler.out.createOutputQueue()
    sent = []
    for group in groups:
        assembler.process(group)
        message = queue.get()
        assert isinstance(message, BoxClouds)
        sent.append(message)
    return sent


def box(x: float, seed: int = 0) -> np.ndarray:
    """Labelled points of a box's top and front, centred at ``x`` mm."""
    return sample_faces(
        Cuboid((x, 0.0, 75.0), (150.0, 150.0, 150.0)), ["-y", "+z"], seed
    )


def test_each_labelled_box_becomes_a_box_cloud_with_its_paired_detection():
    rect = dai.RotatedRect(dai.Rect(dai.Point2f(0.2, 0.3), dai.Point2f(0.6, 0.7)))
    first = box_detection(4, bounding_box=rect, confidence=0.75)
    background = np.full((500, 3), -5000.0, np.float32)
    cloud = label_cloud([paint_view(0, {4: box(-300), 9: box(300)}, background)])

    [clouds] = run([box_group(cloud, [box_detections([first, 9])])])

    by_track = {c.detections[0].track_id: c for c in clouds.items}
    assert sorted(by_track) == [4, 9]
    detection = by_track[4].detections
    assert detection == (
        BoxViewDetection(view=0, track_id=4, bounding_box=rect, confidence=0.75),
    )
    assert by_track[4].points[:, 0].max() < 0 < by_track[9].points[:, 0].min()
    for box_cloud in clouds.items:
        assert (box_cloud.points[:, 2] > -1000).all()


def frame(
    views: list[dict[int, np.ndarray]],
    detections: list[list[int]] | None = None,
    order: list[int] | None = None,
) -> dai.MessageGroup:
    """Group of painted views; each box is detected under its track by default."""
    pixels = [paint_view(view, boxes) for view, boxes in enumerate(views)]
    if order is not None:
        pixels = [pixels[i] for i in order]
    if detections is None:
        detections = [list(boxes) for boxes in views]
    return box_group(label_cloud(pixels), [box_detections(d) for d in detections])


def empty_frame() -> dai.MessageGroup:
    background = np.full((100, 3), 2000.0, np.float32)
    return box_group(label_cloud([paint_view(0, {}, background)]), [box_detections([])])


def summary(clouds: BoxClouds) -> list[tuple[tuple[int, int], float, int]]:
    """Each box cloud's (view, track) key, coverage and box ID, sorted."""
    return sorted(
        (
            (c.detections[0].view, c.detections[0].track_id),
            c.depth_coverage,
            c.box_id,
        )
        for c in clouds.items
    )


def test_the_order_of_the_views_in_the_cloud_does_not_change_keys_coverage_or_ids():
    views = [{1: box(-300, seed=0)}, {2: box(300, seed=1)}]

    in_order = run([frame(views, order=[0, 1]) for _ in range(2)])
    swapped = run([frame(views, order=[1, 0]) for _ in range(2)])

    assert [summary(c) for c in in_order] == [summary(c) for c in swapped]
    assert [key for key, _, _ in summary(in_order[0])] == [(0, 1), (1, 2)]


def test_depth_coverage_counts_valid_labelled_points_before_cleaning():
    valid = box(0)
    valid = valid[np.any(valid != 0, axis=1)]
    valid = valid[: len(valid) // 3 * 3]
    holes = np.zeros((len(valid) // 3, 3), np.float32)

    [clouds] = run([frame([{1: np.concatenate([holes, valid])}])])

    [box_cloud] = clouds.items
    assert box_cloud.depth_coverage == pytest.approx(0.75)
    assert np.any(box_cloud.points != 0, axis=1).all()


def test_a_one_view_box_cloud_holds_its_cleaned_box_view_as_float32():
    raw = box(0)
    valid = raw[np.any(raw != 0, axis=1)]

    [clouds] = run([frame([{1: raw}])])

    [box_cloud] = clouds.items
    assert box_cloud.points.dtype == np.float32
    np.testing.assert_array_equal(
        box_cloud.points, BoxGeometry().clean(valid).astype(np.float32)
    )


def patch_cleaned_to(count: int) -> np.ndarray:
    """Return scattered points that ``BoxGeometry.clean`` thins to ``count``."""
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        points = np.column_stack(
            [rng.uniform(0, 60, (count * 3 // 2, 2)), np.full(count * 3 // 2, 900)]
        ).astype(np.float32)
        if len(BoxGeometry().clean(points)) == count:
            return points
    raise AssertionError(f"no patch cleans to {count} points")


def test_min_points_after_cleaning_keeps_a_box_view_and_one_fewer_makes_it_absent():
    enough = patch_cleaned_to(BoxGeometry.MIN_POINTS)
    too_few = patch_cleaned_to(BoxGeometry.MIN_POINTS - 1)

    sent = run(
        [frame([{1: enough}])]
        + [frame([{1: too_few}])] * MAX_ABSENT_FRAMES
        + [frame([{1: enough}])]
    )

    [first] = sent[0].items
    assert len(first.points) == BoxGeometry.MIN_POINTS
    assert all(len(clouds.items) == 0 for clouds in sent[1:-1])
    [returned] = sent[-1].items
    assert returned.box_id != first.box_id


def box_ids(clouds: BoxClouds) -> dict[tuple[int, int], int]:
    return {
        (c.detections[0].view, c.detections[0].track_id): c.box_id for c in clouds.items
    }


def test_a_new_box_gets_a_new_id_that_stays_and_is_not_its_track_id():
    first, second = box(-300, seed=0), box(300, seed=1)

    sent = run(
        [
            frame([{7: first}]),
            frame([{7: first, 8: second}]),
            frame([{8: second, 7: first}]),
        ]
    )

    ids = [box_ids(clouds) for clouds in sent]
    assert ids[0][(0, 7)] != 7
    assert ids[1] == ids[2]
    assert ids[1][(0, 7)] == ids[0][(0, 7)]
    assert ids[1][(0, 8)] != ids[1][(0, 7)]


@pytest.mark.parametrize(
    ("absent", "keeps_id"),
    [(MAX_ABSENT_FRAMES - 1, True), (MAX_ABSENT_FRAMES, False)],
)
def test_a_key_regains_its_id_only_within_max_absent_frames(absent, keeps_id):
    points = box(0)

    sent = run(
        [frame([{5: points}])] + [empty_frame()] * absent + [frame([{5: points}])]
    )

    assert all(len(clouds.items) == 0 for clouds in sent[1:-1])
    assert (box_ids(sent[-1]) == box_ids(sent[0])) is keeps_id


def test_box_ids_are_never_reused():
    points = box(0)

    sent = run(
        [frame([{1: points}])]
        + [empty_frame()] * MAX_ABSENT_FRAMES
        + [frame([{2: points}]), frame([{1: points, 2: points}])]
    )

    first = box_ids(sent[0])[(0, 1)]
    second = box_ids(sent[-2])[(0, 2)]
    last = box_ids(sent[-1])
    assert last[(0, 2)] == second
    assert len({first, second, last[(0, 1)]}) == 3


def test_a_label_without_a_matching_detection_is_dropped_with_a_warning(caplog):
    sent = run([frame([{4: box(-300), 9: box(300, seed=1)}], detections=[[4, 8]])])

    assert list(box_ids(sent[0])) == [(0, 4)]
    assert "view 0" in caplog.text
    assert "label 9" in caplog.text


def test_a_track_above_the_label_range_keeps_its_id_apart_from_its_label_twin():
    points = box(0)

    sent = run([frame([{70_000: points}])] * 2 + [frame([{4464: points}])])

    first, second, twin = (box_ids(clouds) for clouds in sent)
    assert first == second
    assert list(first) == [(0, 70_000)]
    assert list(twin) == [(0, 4464)]
    assert twin[(0, 4464)] != first[(0, 70_000)]


def test_touching_tracks_of_one_view_never_share_an_id():
    views = two_touching_boxes().views[:1]

    sent = run([frame(views)] * 2)

    assert box_ids(sent[0]) == box_ids(sent[1])
    assert len(set(box_ids(sent[0]).values())) == 2


def test_box_clouds_carry_the_cloud_stamps_not_the_detections():
    cloud = label_cloud([paint_view(0, {1: box(0)})])
    cloud.setTimestamp(timedelta(seconds=3))
    cloud.setTimestampDevice(timedelta(seconds=13))
    cloud.setSequenceNum(42)
    detections = box_detections([1])
    detections.setTimestamp(timedelta(seconds=1))
    detections.setTimestampDevice(timedelta(seconds=11))
    detections.setSequenceNum(7)

    [clouds] = run([box_group(cloud, [detections])])

    assert clouds.getTimestamp() == timedelta(seconds=3)
    assert clouds.getTimestampDevice() == timedelta(seconds=13)
    assert clouds.getSequenceNum() == 42


@pytest.mark.parametrize("scene", [two_separate_boxes, partial_view])
def test_each_scene_associates_into_its_expected_number_of_boxes(scene):
    built = scene()

    [clouds] = run([frame(built.views)])

    assert len(clouds.items) == built.boxes


def keys(box_cloud: BoxCloud) -> set[tuple[int, int]]:
    return {(d.view, d.track_id) for d in box_cloud.detections}


def test_touching_boxes_merge_falsely_through_the_view_that_sees_them_as_one():
    [clouds] = run([frame(two_touching_boxes().views)])

    [box_cloud] = clouds.items
    assert keys(box_cloud) == {(0, 1), (0, 2), (1, 1)}


def test_two_box_views_of_one_view_are_never_linked_directly():
    points = box(0)

    [clouds] = run([frame([{1: points, 2: points.copy()}, {}])])

    assert sorted(keys(c) for c in clouds.items) == [{(0, 1)}, {(0, 2)}]


def flat_patch(
    rng: np.random.Generator, count: int, x: float, size: float = 40
) -> np.ndarray:
    """Scattered points of a square flat patch whose corner is at (``x``, ``x``)."""
    return np.column_stack(
        [
            rng.uniform(x, x + size, (count, 2)),
            np.full(count, 900),
        ]
    ).astype(np.float32)


def near_and_far(shortfall: int) -> np.ndarray:
    """Points whose cleaned box view is ``shortfall`` near points short of exactly
    ``MIN_LINK_FRACTION`` near.

    Near points lie on the patch at x = 0; far ones on patches 200 and 400 mm away.
    """
    for seed in range(5000):
        rng = np.random.default_rng(seed)
        points = np.concatenate([flat_patch(rng, 60, x) for x in (0, 200, 400)])
        cleaned = BoxGeometry().clean(points)
        near = np.count_nonzero(cleaned[:, 0] < 100)
        if (near + shortfall) / len(cleaned) == MIN_LINK_FRACTION:
            return points
    raise AssertionError(f"no patch falls {shortfall} short of linking")


@pytest.mark.parametrize(("shortfall", "linked"), [(0, True), (1, False)])
def test_box_views_link_from_min_link_fraction_of_the_smaller_ones_points(
    shortfall, linked
):
    # Covers the near patch with a 40 mm margin.
    larger = flat_patch(np.random.default_rng(0), 18000, -40, size=120)
    smaller = near_and_far(shortfall)
    assert len(BoxGeometry().clean(larger)) > len(BoxGeometry().clean(smaller))

    [clouds] = run([frame([{1: larger}, {2: smaller}])])

    assert len(clouds.items) == (1 if linked else 2)


# How far inside or outside LINK_DISTANCE_MM the box views lie.
DISTANCE_MARGIN_MM = 0.05


@pytest.fixture
def distance_checks(monkeypatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """The box view pairs whose points went through the distance check."""
    checks = []
    distance_check = box_cloud_assembler._linked

    def spy(first: np.ndarray, second: np.ndarray) -> bool:
        checks.append((first, second))
        return distance_check(first, second)

    monkeypatch.setattr(box_cloud_assembler, "_linked", spy)
    return checks


@pytest.mark.parametrize(("offset", "linked"), [(-1, True), (1, False)])
def test_box_views_link_only_within_link_distance(distance_checks, offset, linked):
    """Box views whose bounds lie over ``LINK_DISTANCE_MM`` apart skip the distance
    check."""
    points = flat_patch(np.random.default_rng(0), 2000, 0)
    lift = np.array([0, 0, LINK_DISTANCE_MM + offset * DISTANCE_MARGIN_MM], np.float32)

    [clouds] = run([frame([{1: points}, {2: points + lift}])])

    assert len(clouds.items) == (1 if linked else 2)
    assert len(distance_checks) == (1 if linked else 0)


@pytest.mark.parametrize(("offset", "linked"), [(-1, True), (1, False)])
def test_box_views_with_overlapping_bounds_link_only_within_link_distance(
    distance_checks, offset, linked
):
    rng = np.random.default_rng(0)
    points = flat_patch(rng, 2000, 0)
    lift = np.array([0, 0, LINK_DISTANCE_MM + offset * DISTANCE_MARGIN_MM], np.float32)
    # A far patch at the lifted height makes the bounds overlap, so the distance
    # between the patch and its lifted copy decides.
    far = flat_patch(rng, 2000, 200) + lift

    [clouds] = run([frame([{1: np.concatenate([points, far])}, {2: points + lift}])])

    assert len(clouds.items) == (1 if linked else 2)
    assert len(distance_checks) == 1


def test_a_two_view_box_cloud_keeps_one_float32_point_per_merge_voxel():
    views = opposite_sides().views

    [clouds] = run([frame(views)])

    [box_cloud] = clouds.items
    assert box_cloud.points.dtype == np.float32
    assert box_cloud.points.shape[1] == 3
    voxels = np.floor(box_cloud.points / MERGE_VOXEL_SIZE_MM)
    assert len(np.unique(voxels, axis=0)) == len(box_cloud.points)
    cleaned = sum(
        len(BoxGeometry().clean(v[1][np.any(v[1] != 0, axis=1)])) for v in views
    )
    assert len(box_cloud.points) < cleaned


def with_holes(points: np.ndarray, coverage: float) -> np.ndarray:
    """The box view's valid points plus holes, at the given coverage."""
    valid = points[np.any(points != 0, axis=1)]
    holes = round(len(valid) * (1 - coverage) / coverage)
    return np.concatenate([np.zeros((holes, 3), np.float32), valid])


def test_a_box_clouds_coverage_is_the_highest_of_its_box_views():
    views = opposite_sides().views

    [clouds] = run(
        [frame([{1: with_holes(views[0][1], 0.5)}, {1: with_holes(views[1][1], 0.8)}])]
    )

    [box_cloud] = clouds.items
    assert box_cloud.depth_coverage == pytest.approx(0.8, abs=0.01)


def test_a_dropped_box_view_does_not_count_toward_coverage():
    too_few = patch_cleaned_to(BoxGeometry.MIN_POINTS - 1)
    rng = np.random.default_rng(0)
    kept = with_holes(flat_patch(rng, 2000, 0), 0.5)

    [clouds] = run([frame([{1: kept}, {2: too_few}])])

    [box_cloud] = clouds.items
    assert keys(box_cloud) == {(0, 1)}
    assert box_cloud.depth_coverage == pytest.approx(0.5, abs=0.01)


def ids_by_key(clouds: BoxClouds) -> dict[tuple[int, int], int]:
    """Every key's box ID, over all box views of every box cloud."""
    return {key: c.box_id for c in clouds.items for key in keys(c)}


def test_a_merge_keeps_the_older_id_and_the_newer_never_reappears():
    left, right = opposite_sides().views

    sent = run(
        [
            frame([left, {}]),
            frame([{}, right]),
            frame([left, right]),
            frame([{}, right]),
        ]
    )

    older = ids_by_key(sent[0])[(0, 1)]
    newer = ids_by_key(sent[1])[(1, 1)]
    assert older != newer
    assert ids_by_key(sent[2]) == {(0, 1): older, (1, 1): older}
    assert ids_by_key(sent[3]) == {(1, 1): older}


def test_on_a_split_the_group_with_most_of_the_ids_keys_keeps_it():
    merged = two_touching_boxes()
    back = [
        sample_faces(Cuboid((x, 0.0, 75.0), (150.0, 150.0, 150.0)), ["+y", "+z"], seed)
        for x, seed in ((-75.0, 2), (75.0, 3))
    ]

    sent = run(
        [frame(merged.views), frame([merged.views[0], {1: back[0], 3: back[1]}])]
    )

    [box_id] = set(ids_by_key(sent[0]).values())
    after = ids_by_key(sent[1])
    assert after[(0, 1)] == after[(1, 1)] == box_id
    assert after[(0, 2)] == after[(1, 3)] != box_id


def test_a_track_joining_a_box_inherits_its_id():
    left, right = different_track_ids().views

    sent = run([frame([left, {}]), frame([left, right])])

    [box_id] = ids_by_key(sent[0]).values()
    assert ids_by_key(sent[1]) == {(0, 3): box_id, (1, 8): box_id}


def test_a_box_keeps_its_id_through_one_view_while_the_other_loses_its_track():
    left, right = opposite_sides().views

    sent = run([frame([left, right]), frame([left, {}]), frame([left, right])])

    [box_id] = set(ids_by_key(sent[0]).values())
    assert ids_by_key(sent[1]) == {(0, 1): box_id}
    assert ids_by_key(sent[2]) == {(0, 1): box_id, (1, 1): box_id}


@pytest.mark.parametrize(
    ("points", "winner"),
    [([100, 200], 1), ([200, 100], 0), ([100, 100], 0)],
)
def test_an_id_tied_on_keys_goes_to_more_points_then_to_the_first_group(points, winner):
    claims = [_Claim(candidates=[7], points=count) for count in points]

    assert _choose_group(7, claims) == winner


def test_an_id_goes_to_the_unassigned_group_with_most_of_its_keys():
    claims = [
        _Claim(candidates=[7], points=900),
        _Claim(candidates=[7, 7], points=1),
        _Claim(candidates=[7, 7, 7], points=1, box_id=3),
    ]

    assert _choose_group(7, claims) == 1
