from datetime import timedelta
from typing import cast

import depthai as dai
import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import (  # noqa: E402
    BoxDetection,
    BoxDetections,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
    SupportPlane,
)
from depthai_nodes.node import BoxDimensioner  # noqa: E402
from tests.utils import OutputMock  # noqa: E402

CLOUD_SIZE = (64, 48)
FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


def output() -> dai.Node.Output:
    return cast(dai.Node.Output, OutputMock())


def build(size=CLOUD_SIZE, plane=None) -> BoxDimensioner:
    return BoxDimensioner().build(output(), output(), size, plane)


def measurements(queue) -> BoxMeasurements:
    message = queue.get()
    assert isinstance(message, BoxMeasurements)
    return message


def height(item: BoxMeasurementResult) -> float:
    assert item.dimensions is not None
    return item.dimensions.height_mm


def point_cloud(queue) -> dai.PointCloudData:
    message = queue.get()
    assert isinstance(message, dai.PointCloudData)
    return message


def stamp(message, seq: int):
    message.setSequenceNum(seq)
    message.setTimestamp(timedelta(seconds=seq))
    message.setTimestampDevice(timedelta(seconds=seq + 10))
    return message


def flat_top_cloud(seq=1, height=100.0, depth=True, coverage=1.0):
    """An organized cloud of a plane ``height`` mm above a floor at 1000 mm.

    Without ``depth`` every point is invalid; with ``coverage`` below one, every
    fifth point is.
    """
    width, rows = CLOUD_SIZE
    x, y = np.meshgrid(np.linspace(-200, 200, width), np.linspace(-150, 150, rows))
    points = np.column_stack(
        (x.ravel(), y.ravel(), np.full(width * rows, 1000.0 - height))
    )
    if not depth:
        points[:] = 0
    elif coverage < 1:
        points[::5] = 0
    cloud = dai.PointCloudData()
    cloud.setPoints(points.astype(np.float32))  # pyright: ignore[reportArgumentType]
    return stamp(cloud, seq)


def box(track_id=7, bounds=(0.2, 0.2, 0.8, 0.8)):
    xmin, ymin, xmax, ymax = bounds
    rect = dai.RotatedRect(
        dai.Point2f((xmin + xmax) / 2, (ymin + ymax) / 2),
        dai.Size2f(xmax - xmin, ymax - ymin),
        0,
    )
    return BoxDetection(
        track_id=track_id,
        bounding_box=rect,
        confidence=0.8,
        track_status=dai.Tracklet.TrackingStatus.TRACKED,
    )


def centered_mask(index=0):
    mask = np.full((24, 32), 255, dtype=np.uint8)
    mask[5:19, 7:25] = index
    return mask


class Dimensioning:
    """A BoxDimensioner built on mock outputs, fed one frame at a time."""

    def __init__(self, plane: SupportPlane | None = FLOOR):
        self.dimensioner = BoxDimensioner().build(
            detections=output(),
            pointCloud=output(),
            pointCloudSize=CLOUD_SIZE,
            supportPlane=plane,
        )
        self.output = self.dimensioner.out.createOutputQueue()
        self.cloud_output = self.dimensioner.boxCloudOutput.createOutputQueue()
        self.seq = 0

    def send(self, items=None, mask=None, height=100, depth=True, coverage=1.0):
        """Process one frame; by default one centered box over a flat top."""
        self.seq += 1
        if items is None:
            items, mask = [box()], centered_mask()
        if mask is None:
            mask = np.full((24, 32), 255, dtype=np.uint8)
        cloud = flat_top_cloud(self.seq, height, depth, coverage)
        self.dimensioner.process(BoxDetections(items, mask), cloud)
        return cloud

    def receive(self) -> BoxMeasurements:
        return measurements(self.output)

    def item(self, **kwargs):
        self.send(**kwargs)
        (item,) = self.receive().items
        return item


def test_measured_box_carries_detection_fields_and_dimensions():
    item = Dimensioning().item()

    assert item.measurement_status == MeasurementStatus.MEASURED
    assert height(item) == pytest.approx(100, abs=0.05)
    assert item.track_id == 7
    assert item.confidence == pytest.approx(0.8)
    assert item.track_status == dai.Tracklet.TrackingStatus.TRACKED
    assert item.bounding_box.center.x == pytest.approx(0.5)
    assert item.bounding_box.size.width == pytest.approx(0.6)


def test_measurements_carry_the_cloud_timing():
    app = Dimensioning()
    cloud = app.send(items=[])

    message = app.receive()

    assert message.items == []
    assert message.getSequenceNum() == cloud.getSequenceNum()
    assert message.getTimestamp() == cloud.getTimestamp()
    assert message.getTimestampDevice() == cloud.getTimestampDevice()


@pytest.mark.parametrize("size", [(0, 48), (64, -1), (64,), (64, 48, 1)])
def test_build_rejects_cloud_sizes_without_two_positive_values(size):
    with pytest.raises(ValueError):
        build(size)


@pytest.mark.parametrize(
    "size", [(64.0, 48), (True, True), (64, True), ("64", 48), (None, 48), [64, 48]]
)
def test_build_rejects_cloud_sizes_that_are_not_integer_pairs(size):
    with pytest.raises(TypeError):
        build(size)


def test_build_accepts_numpy_integer_cloud_size():
    size = (np.int64(64), np.int32(48))
    dimensioner = build(size)
    queue = dimensioner.out.createOutputQueue()

    dimensioner.process(BoxDetections([box()], centered_mask()), flat_top_cloud())

    assert (
        measurements(queue).items[0].measurement_status == MeasurementStatus.INELIGIBLE
    )


def test_build_rejects_a_plane_that_is_not_a_support_plane():
    with pytest.raises(TypeError):
        build(plane=(0, 0, 1, -1000))


def test_second_build_is_rejected():
    dimensioner = build()
    with pytest.raises(RuntimeError):
        dimensioner.build(output(), output(), CLOUD_SIZE)


def test_plane_methods_work_before_build():
    dimensioner = BoxDimensioner()
    assert dimensioner.getSupportPlane() is None
    dimensioner.setSupportPlane(FLOOR)
    assert dimensioner.getSupportPlane() == FLOOR
    dimensioner.clearSupportPlane()
    assert dimensioner.getSupportPlane() is None
    with pytest.raises(TypeError):
        dimensioner.setSupportPlane((0, 0, 1, -1000))  # pyright: ignore[reportArgumentType]


def test_without_plane_boxes_are_ineligible_without_dimensions():
    item = Dimensioning(plane=None).item()

    assert item.measurement_status == MeasurementStatus.INELIGIBLE
    assert item.dimensions is None


def test_border_box_is_ineligible_and_keeps_the_aggregate():
    app = Dimensioning()
    first = app.item()
    for bounds in [(0.04, 0.2, 0.8, 0.8), (0.2, 0.2, 0.8, 0.96)]:
        border = app.item(items=[box(bounds=bounds)], mask=centered_mask())
        assert border.measurement_status == MeasurementStatus.INELIGIBLE
        assert border.dimensions == first.dimensions
    inside = app.item(items=[box(bounds=(0.06, 0.2, 0.8, 0.8))], mask=centered_mask())
    assert inside.measurement_status == MeasurementStatus.MEASURED


@pytest.mark.parametrize("kwargs", [{"depth": False}, {"coverage": 0.8}])
def test_too_few_valid_points_under_the_mask_fail_and_keep_the_aggregate(kwargs):
    app = Dimensioning()
    first = app.item()

    failed = app.item(**kwargs)

    assert failed.measurement_status == MeasurementStatus.FAILED
    assert failed.dimensions == first.dimensions


def test_failed_box_without_samples_has_no_dimensions():
    failed = Dimensioning().item(height=0)

    assert failed.measurement_status == MeasurementStatus.FAILED
    assert failed.dimensions is None


def test_aggregate_is_the_median_of_samples():
    app = Dimensioning()
    *_, item = [app.item(height=height) for height in (50, 150, 100)]

    assert height(item) == pytest.approx(100, abs=0.05)


def test_only_the_last_600_samples_contribute():
    app = Dimensioning()
    *_, item = [app.item(height=h) for h in [50] * 301 + [150] * 300]

    assert height(item) == pytest.approx(100, abs=0.05)


def test_tracks_are_aggregated_independently():
    app = Dimensioning()
    mask = np.full((24, 32), 255, dtype=np.uint8)
    mask[5:19, 4:14] = 0
    mask[5:19, 18:28] = 1

    app.send(
        items=[box(1, (0.1, 0.2, 0.45, 0.8)), box(2, (0.55, 0.2, 0.9, 0.8))],
        mask=mask,
    )
    first, second = app.receive().items

    assert (first.track_id, second.track_id) == (1, 2)
    assert first.measurement_status == MeasurementStatus.MEASURED
    assert second.measurement_status == MeasurementStatus.MEASURED


def test_absent_track_samples_survive_29_frames_and_drop_at_30():
    app = Dimensioning()
    app.item(height=100)
    for _ in range(29):
        app.send(items=[])
        assert app.receive().items == []
    kept = app.item(depth=False)
    assert height(kept) == pytest.approx(100, abs=0.05)
    for _ in range(30):
        app.send(items=[])
        app.receive()

    reappeared = app.item(depth=False)
    assert reappeared.dimensions is None
    fresh = app.item(height=70)
    assert height(fresh) == pytest.approx(70, abs=0.05)


def test_setting_or_clearing_the_plane_resets_samples():
    app = Dimensioning()
    app.item(height=100)
    app.dimensioner.setSupportPlane(SupportPlane(normal=(0, 0, 1), offset_mm=-1200))
    moved = app.item()
    assert height(moved) == pytest.approx(300, abs=0.05)
    app.dimensioner.setSupportPlane(FLOOR)
    assert app.item(depth=False).dimensions is None
    app.item(height=100)

    app.dimensioner.clearSupportPlane()
    cleared = app.item()

    assert cleared.measurement_status == MeasurementStatus.INELIGIBLE
    assert cleared.dimensions is None


def test_sign_equivalent_planes_measure_the_same_height():
    app = Dimensioning(SupportPlane(normal=(0, 0, -1), offset_mm=1000))

    item = app.item(height=100)

    assert item.measurement_status == MeasurementStatus.MEASURED
    assert height(item) == pytest.approx(100, abs=0.05)


def test_box_cloud_holds_valid_points_under_masks_with_cloud_timing():
    app = Dimensioning()
    cloud = app.send()

    box_cloud = point_cloud(app.cloud_output)

    points = np.asarray(box_cloud.getPoints())
    # Mask rows 5:19 of 24 and columns 7:25 of 32 cover 28 x 36 cloud cells.
    assert points.shape == (28 * 36, 3)
    np.testing.assert_allclose(points[:, 2], 900)
    assert box_cloud.getSequenceNum() == cloud.getSequenceNum()
    assert box_cloud.getTimestamp() == cloud.getTimestamp()
    assert box_cloud.getTimestampDevice() == cloud.getTimestampDevice()


def test_box_cloud_is_empty_without_valid_points():
    app = Dimensioning()
    app.send(depth=False)

    assert np.asarray(point_cloud(app.cloud_output).getPoints()).size == 0


def test_unorganized_cloud_yields_boxes_without_dimensions():
    app = Dimensioning()
    cloud = dai.PointCloudData()
    cloud.setPoints(np.full((10, 3), 900, dtype=np.float32))  # pyright: ignore[reportArgumentType]

    app.dimensioner.process(BoxDetections([box()], centered_mask()), stamp(cloud, 1))
    (item,) = app.receive().items

    assert item.measurement_status == MeasurementStatus.FAILED
    assert item.dimensions is None


def test_process_rejects_messages_of_the_wrong_type():
    app = Dimensioning()
    with pytest.raises(TypeError):
        app.dimensioner.process(flat_top_cloud(), flat_top_cloud())


# Area just below (0.4996) and just above (0.5004) half of the frame.
JUST_BELOW_HALF = (0.1, 0.1, 0.9, 0.7245)
JUST_ABOVE_HALF = (0.1, 0.1, 0.9, 0.7255)


def test_detection_just_below_half_the_frame_is_measured():
    item = Dimensioning().item(
        items=[box(bounds=JUST_BELOW_HALF)], mask=centered_mask()
    )

    assert item.measurement_status == MeasurementStatus.MEASURED


def test_detection_just_above_half_the_frame_is_omitted():
    app = Dimensioning()
    app.send(items=[box(bounds=JUST_ABOVE_HALF)], mask=centered_mask())

    assert app.receive().items == []


def test_track_oversized_in_some_frames_keeps_samples_from_other_frames():
    app = Dimensioning()
    first = app.item(height=100)
    for _ in range(40):
        app.send(items=[box(bounds=JUST_ABOVE_HALF)], mask=centered_mask(), height=300)
        assert app.receive().items == []

    item = app.item(depth=False)

    assert item.track_id == 7
    assert item.dimensions == first.dimensions
