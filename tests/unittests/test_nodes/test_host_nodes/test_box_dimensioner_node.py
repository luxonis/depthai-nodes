from datetime import timedelta
from typing import cast

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    BoxCloud,
    BoxClouds,
    BoxMeasurementResult,
    BoxMeasurements,
    BoxViewDetection,
    MeasurementStatus,
    SupportPlane,
)
from depthai_nodes.node import BoxDimensioner
from depthai_nodes.node.box_dimensioner.box_dimensioner import (
    MAX_ABSENT_FRAMES,
    MAX_SAMPLES_PER_BOX,
    MIN_VALID_POINT_RATIO,
)
from depthai_nodes.node.box_dimensioner.box_geometry import BoxGeometry
from tests.utils import OutputMock
from tests.utils.messages.creators.box import (
    create_box_view_detection,
    create_rotated_rect,
)

FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)
JUST_BELOW_RATIO = float(np.nextafter(MIN_VALID_POINT_RATIO, 0))


def output() -> dai.Node.Output:
    return cast(dai.Node.Output, OutputMock())


def build(plane=None) -> BoxDimensioner:
    return BoxDimensioner().build(output(), plane)


def height(item: BoxMeasurementResult) -> float:
    assert item.dimensions is not None
    return item.dimensions.height_mm


def flat_top(height=100.0) -> np.ndarray:
    """Unorganized points of a plane ``height`` mm above a floor at 1000 mm."""
    x, y = np.meshgrid(np.linspace(-200, 200, 64), np.linspace(-150, 150, 48))
    return np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, 1000.0 - height))
    ).astype(np.float32)


def box_cloud(box_id=7, height=100.0, coverage=1.0, points=None, detections=None):
    return BoxCloud(
        box_id=box_id,
        points=flat_top(height) if points is None else points,
        depth_coverage=coverage,
        detections=detections or (create_box_view_detection(track_id=box_id),),
    )


class Dimensioning:
    """A BoxDimensioner built on a mock output, fed one frame at a time."""

    def __init__(self, plane: SupportPlane | None = FLOOR):
        self.dimensioner = BoxDimensioner().build(
            boxClouds=output(), supportPlane=plane
        )
        self.output = self.dimensioner.out.createOutputQueue()
        self.seq = 0

    def send(self, clouds=None, height=100, coverage=1.0, box_id=7) -> BoxClouds:
        """Process one frame; by default one box cloud over a flat top."""
        self.seq += 1
        if clouds is None:
            clouds = [box_cloud(box_id, height, coverage)]
        message = BoxClouds(clouds)
        message.setSequenceNum(self.seq)
        message.setTimestamp(timedelta(seconds=self.seq))
        message.setTimestampDevice(timedelta(seconds=self.seq + 10))
        self.dimensioner.process(message)
        return message

    def receive(self) -> BoxMeasurements:
        message = self.output.get()
        assert isinstance(message, BoxMeasurements)
        return message

    def item(self, **kwargs) -> BoxMeasurementResult:
        self.send(**kwargs)
        (item,) = self.receive().items
        return item


def test_measured_box_carries_its_box_id_detections_and_dimensions():
    detections = (
        BoxViewDetection(
            view=1,
            track_id=4,
            bounding_box=create_rotated_rect((0.3, 0.6), (0.2, 0.1)),
            confidence=0.75,
        ),
        create_box_view_detection(view=0, track_id=9),
    )

    app = Dimensioning()
    app.send(clouds=[box_cloud(box_id=12, detections=detections)])
    (item,) = app.receive().items

    assert item.measurement_status == MeasurementStatus.MEASURED
    assert height(item) == pytest.approx(100, abs=0.05)
    assert item.box_id == 12
    assert item.detections == detections


def test_measurements_carry_the_box_clouds_stamps():
    app = Dimensioning()
    clouds = app.send(clouds=[])

    message = app.receive()

    assert message.items == []
    assert message.getSequenceNum() == clouds.getSequenceNum()
    assert message.getTimestamp() == clouds.getTimestamp()
    assert message.getTimestampDevice() == clouds.getTimestampDevice()


def test_build_rejects_a_plane_that_is_not_a_support_plane():
    with pytest.raises(TypeError):
        build(plane=(0, 0, 1, -1000))


def test_second_build_is_rejected():
    dimensioner = build()
    with pytest.raises(RuntimeError):
        dimensioner.build(output())


def test_plane_methods_work_before_build():
    dimensioner = BoxDimensioner()
    assert dimensioner.getSupportPlane() is None
    dimensioner.setSupportPlane(FLOOR)
    assert dimensioner.getSupportPlane() == FLOOR
    dimensioner.clearSupportPlane()
    assert dimensioner.getSupportPlane() is None
    with pytest.raises(TypeError):
        dimensioner.setSupportPlane((0, 0, 1, -1000))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("coverage", [1.0, JUST_BELOW_RATIO])
def test_without_plane_boxes_are_ineligible_without_dimensions(coverage):
    item = Dimensioning(plane=None).item(coverage=coverage)

    assert item.measurement_status == MeasurementStatus.INELIGIBLE
    assert item.dimensions is None


def test_coverage_at_the_minimum_ratio_is_measured():
    item = Dimensioning().item(coverage=MIN_VALID_POINT_RATIO)

    assert item.measurement_status == MeasurementStatus.MEASURED


@pytest.mark.parametrize(
    "cloud",
    [
        box_cloud(coverage=JUST_BELOW_RATIO),
        box_cloud(points=flat_top()[: BoxGeometry.MIN_POINTS - 1]),
    ],
    ids=["coverage-below-ratio", "too-few-points"],
)
def test_unmeasurable_box_fails_and_keeps_the_aggregate(cloud):
    app = Dimensioning()
    first = app.item()

    failed = app.item(clouds=[cloud])

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


def test_only_the_last_max_samples_contribute():
    half = MAX_SAMPLES_PER_BOX // 2
    app = Dimensioning()
    *_, item = [app.item(height=h) for h in [50] * (half + 1) + [150] * half]

    assert height(item) == pytest.approx(100, abs=0.05)


def test_boxes_are_aggregated_independently():
    app = Dimensioning()
    app.send(clouds=[box_cloud(1, height=80), box_cloud(2, height=120)])
    app.receive()

    app.send(
        clouds=[
            box_cloud(1, coverage=JUST_BELOW_RATIO),
            box_cloud(2, coverage=JUST_BELOW_RATIO),
        ]
    )
    first, second = app.receive().items

    assert (first.box_id, second.box_id) == (1, 2)
    assert height(first) == pytest.approx(80, abs=0.05)
    assert height(second) == pytest.approx(120, abs=0.05)


def test_a_new_box_id_starts_empty_even_with_another_boxs_detections():
    detections = (create_box_view_detection(view=0, track_id=7),)
    app = Dimensioning()
    app.item(clouds=[box_cloud(box_id=7, detections=detections)])

    new = app.item(
        clouds=[box_cloud(box_id=8, coverage=JUST_BELOW_RATIO, detections=detections)]
    )

    assert new.measurement_status == MeasurementStatus.FAILED
    assert new.dimensions is None


def test_absent_box_samples_survive_until_max_absent_frames():
    app = Dimensioning()
    app.item(height=100)
    for _ in range(MAX_ABSENT_FRAMES - 1):
        app.send(clouds=[])
        assert app.receive().items == []
    kept = app.item(coverage=JUST_BELOW_RATIO)
    assert height(kept) == pytest.approx(100, abs=0.05)
    for _ in range(MAX_ABSENT_FRAMES):
        app.send(clouds=[])
        app.receive()

    reappeared = app.item(coverage=JUST_BELOW_RATIO)
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
    assert app.item(coverage=JUST_BELOW_RATIO).dimensions is None
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


def test_process_rejects_messages_of_the_wrong_type():
    app = Dimensioning()
    with pytest.raises(TypeError):
        app.dimensioner.process(dai.PointCloudData())
