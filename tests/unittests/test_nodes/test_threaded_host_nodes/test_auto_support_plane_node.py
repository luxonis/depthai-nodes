"""`AutoSupportPlane` with its real inner calibrator, both on mock DepthAI queues.

The harness pipeline creates the wrapper and its calibrator subnode on bounded mock
queues. Internal outputs deliver to the inputs they are linked to, and a send that would
block in DepthAI fails the test. One harness pass runs one calibrator pass, then one
wrapper pass.
"""

from collections import deque
from datetime import timedelta
from typing import Any, Callable, cast
from unittest.mock import patch

import depthai as dai
import numpy as np
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import (  # noqa: E402
    AutoSupportPlaneResult,
    SupportPlane,
    SupportPlaneCandidate,
)
from depthai_nodes.node import (  # noqa: E402
    AutoSupportPlane,
    AutoSupportPlaneConfig,
    PlaneCalibrationConfig,
    SupportPlaneCalibrator,
)
from depthai_nodes.node.support_plane_calibrator.support_plane_calibrator import (  # noqa: E402
    IMU_MAX_AGE_SECONDS,
)
from depthai_nodes.node.support_plane_calibrator.support_plane_fitter import (  # noqa: E402
    SupportPlaneFitter,
)
from tests.utils import OutputMock, PipelineMock  # noqa: E402
from tests.utils.nodes.mocks.queue import OutputQueueMock  # noqa: E402
from tests.utils.scripted_open3d import ScriptedScene  # noqa: E402

CLOUD_SIZE = (20, 20)
# DepthAI's default input queue size.
DEFAULT_QUEUE_SIZE = 3


class BoundedInput:
    """An input queue with DepthAI's size and blocking semantics.

    A blocking input that is full fails the test on the next send, because DepthAI would
    block the sender there. A non-blocking input drops its oldest message.
    """

    def __init__(self, name, blocking, queue_size):
        self.name = name
        self._blocking = blocking
        self._max_size = queue_size
        self._messages = deque()

    def setBlocking(self, blocking):
        self._blocking = blocking

    def setMaxSize(self, size):
        self._max_size = size

    def getMaxSize(self):
        return self._max_size

    def send(self, message):
        if len(self._messages) >= self._max_size:
            if self._blocking:
                pytest.fail(
                    f"send to full blocking input {self.name!r} "
                    f"(size {self._max_size}) would block"
                )
            self._messages.popleft()
        self._messages.append(message)

    def tryGet(self):
        return self._messages.popleft() if self._messages else None


class LinkedOutput(OutputMock):
    """An output that delivers to the inputs it is linked to and to output queues."""

    def send(self, message):
        super().send(message)
        for target in self.links:
            target.send(message)


class OnHarnessQueues:
    """Puts a host node on harness queues, running a set number of loop passes."""

    run: Callable[[], None]

    def __init__(self, pipeline):
        self._passes = 0
        self._pipeline = pipeline
        super().__init__()

    def getParentPipeline(self):
        return self._pipeline

    def createInput(
        self, name=None, blocking=True, queueSize=DEFAULT_QUEUE_SIZE, **kwargs
    ):
        return BoundedInput(name, blocking, queueSize)

    def createOutput(self, *args, **kwargs):
        return LinkedOutput()

    def isRunning(self):
        self._passes -= 1
        return self._passes >= 0

    def runPasses(self, passes=1):
        self._passes = passes
        self.run()


class HarnessCalibrator(OnHarnessQueues, SupportPlaneCalibrator):
    pass


class HarnessAutoSupportPlane(OnHarnessQueues, AutoSupportPlane):
    pass


class HarnessPipeline(PipelineMock):
    """Creates the support plane nodes on harness queues."""

    HARNESS_TYPES = {
        SupportPlaneCalibrator: HarnessCalibrator,
        AutoSupportPlane: HarnessAutoSupportPlane,
    }

    def create(self, node_type: Any) -> Any:
        harness_type = self.HARNESS_TYPES.get(node_type)
        if harness_type is None:
            return super().create(node_type)
        node = harness_type(self)
        self._nodes.append(node)
        return node

    def node(self, node_type):
        (node,) = [n for n in self.getAllNodes() if isinstance(n, node_type)]
        return node


class Source(OutputMock):
    """An upstream output; messages sent to it reach the input it is linked to."""

    def __init__(self):
        super().__init__()
        self._target = None

    def link(self, node):
        self._target = node

    def send(self, message):
        assert self._target is not None, "source is not linked"
        self._target.send(message)


class Harness:
    def __init__(self, config=None, size=CLOUD_SIZE, **build_kwargs):
        self.size = size
        self.clouds = Source()
        self.imus = Source()
        self.pipeline = HarnessPipeline()
        self.node: HarnessAutoSupportPlane = self.pipeline.create(AutoSupportPlane)
        self.node.build(
            cast(dai.Node.Output, self.clouds),
            cast(dai.Node.Output, self.imus),
            size,
            config,
            **build_kwargs,
        )
        self.calibrator = self.pipeline.node(HarnessCalibrator)
        self.results = cast(OutputQueueMock, self.node.out.createOutputQueue())
        self._sequence_num = 0

    def gravity(self, x=0.0, y=0.0, z=-9.8):
        message = dai.IMUData()
        packet = dai.IMUPacket()
        packet.acceleroMeter.x = x
        packet.acceleroMeter.y = y
        packet.acceleroMeter.z = z
        message.packets = [packet]
        self.imus.send(message)

    def plane_cloud(self, depth=1000.0):
        """Send a flat organized cloud ``depth`` mm in front of the camera.

        Returns:
            The sent cloud, carrying unique stamps.
        """
        width, height = self.size
        return self.depth_cloud(np.full((height, width), depth))

    def depth_cloud(self, depths):
        """Send an organized cloud with one ``z`` per point, rows by columns, in mm.

        Points sit on a 40 mm grid in ``x`` and ``y``; ``z = 0`` is no depth.

        Returns:
            The sent cloud, carrying unique stamps.
        """
        height, width = depths.shape
        x, y = np.meshgrid(np.arange(width) * 40.0, np.arange(height) * 40.0)
        points = np.column_stack([x.ravel(), y.ravel(), depths.ravel()])
        message = dai.PointCloudData()
        message.setPoints(points.astype(np.float32))  # pyright: ignore[reportArgumentType]
        self._sequence_num += 1
        message.setSequenceNum(self._sequence_num)
        message.setTimestamp(timedelta(seconds=self._sequence_num))
        message.setTimestampDevice(timedelta(seconds=100 + self._sequence_num))
        self.clouds.send(message)
        return message

    def run(self, passes=1):
        for _ in range(passes):
            self.calibrator.runPasses(1)
            self.node.runPasses(1)

    def result(self) -> AutoSupportPlaneResult:
        result = self.results.tryGet()
        assert isinstance(result, AutoSupportPlaneResult)
        return result

    def no_result(self) -> bool:
        return self.results.is_empty()


class Clock:
    """A stand-in for ``time.monotonic`` that moves only when advanced."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    clock = Clock()
    with (
        patch(
            "depthai_nodes.node.support_plane_calibrator.support_plane_calibrator"
            ".time.monotonic",
            clock,
        ),
        patch(
            "depthai_nodes.node.auto_support_plane.auto_support_plane.time.monotonic",
            clock,
        ),
    ):
        yield clock


def scripted(scene):
    return patch("open3d.geometry.PointCloud", scene.cloud)


def floor(offset_mm):
    return SupportPlane(normal=(0, 0, 1), offset_mm=offset_mm)


def scripted_floors(*floors):
    """A scene fitting level floors, each given as ``(offset_mm, inlier_count)``."""
    return ScriptedScene([([0.0, 0.0, 1.0, offset], count) for offset, count in floors])


def on_start(**calibration):
    return AutoSupportPlaneConfig(
        mode=AutoSupportPlaneConfig.Mode.ON_START,
        calibration=PlaneCalibrationConfig(max_plane_passes=3, **calibration),
    )


def assert_stamped_from(result, cloud):
    assert result.getSequenceNum() == cloud.getSequenceNum()
    assert result.getTimestamp() == cloud.getTimestamp()
    assert result.getTimestampDevice() == cloud.getTimestampDevice()


def test_first_fit_sends_the_plane_stamped_from_its_cloud_and_holds_it():
    node = Harness(on_start())
    node.gravity()
    with scripted(scripted_floors((-1000.0, 200))):
        cloud = node.plane_cloud()
        node.run(2)

    result = node.result()
    assert result.passed is True
    assert result.plane == floor(-1000)
    assert_stamped_from(result, cloud)
    for _ in range(5):
        node.gravity()
        node.plane_cloud(depth=1500)
        node.run()
    assert node.no_result()


def reversed_ranking():
    """Reverse the fitter's candidate order, moving its first candidate last."""
    target = (
        "depthai_nodes.node.support_plane_calibrator.support_plane_fitter"
        ".SupportPlaneFitter._rank"
    )
    original = SupportPlaneFitter._rank
    return patch(target, staticmethod(lambda fits, size: original(fits, size)[::-1]))


def first_fit(node, scene):
    """Run passes until the first attempt fits one cloud of ``scene``."""
    node.gravity()
    with scripted(scene), reversed_ranking():
        node.plane_cloud()
        node.run(2)
    return node.result()


def test_default_selection_takes_the_most_inliers_wherever_it_sits():
    node = Harness(on_start())

    result = first_fit(node, scripted_floors((-1000.0, 200), (-1500.0, 100)))

    assert result.plane == floor(-1000)


def test_default_selection_breaks_inlier_ties_by_the_nearest_plane():
    node = Harness(on_start())

    result = first_fit(node, scripted_floors((-1500.0, 150), (-1000.0, 150)))

    assert result.plane == floor(-1000)


def fit_empty_cloud(node):
    """Send a cloud without depth with fresh IMU and run until it is fitted."""
    node.gravity()
    cloud = node.plane_cloud(depth=0)
    node.run()
    return cloud


def test_capped_attempt_sends_no_plane_and_gives_up_until_rebuilt():
    node = Harness(on_start(max_calibration_attempts=2))
    node.run()
    fit_empty_cloud(node)
    assert node.no_result()

    last = fit_empty_cloud(node)

    result = node.result()
    assert result.passed is False
    assert result.plane is None
    assert_stamped_from(result, last)
    for _ in range(5):
        node.gravity()
        node.plane_cloud()
        node.run()
    assert node.no_result()


def test_nothing_is_sent_before_a_cloud_with_fresh_imu(clock):
    node = Harness(on_start())
    node.run(3)
    node.gravity()
    node.run()
    clock.now += IMU_MAX_AGE_SECONDS + 0.01
    node.plane_cloud()
    node.run(3)
    assert node.no_result()

    node.gravity()
    node.run()

    assert node.result().plane == floor(-1000)


def test_uncapped_search_sends_nothing_until_a_plane_is_found():
    node = Harness(on_start(max_calibration_attempts=None))
    node.run()
    for _ in range(10):
        fit_empty_cloud(node)
    assert node.no_result()

    node.gravity()
    node.plane_cloud()
    node.run()

    assert node.result().plane == floor(-1000)


def build_args():
    output = cast(dai.Node.Output, OutputMock())
    return output, output, CLOUD_SIZE


def test_build_rejects_a_config_of_the_wrong_type():
    node = HarnessPipeline().create(AutoSupportPlane)

    with pytest.raises(TypeError):
        node.build(*build_args(), PlaneCalibrationConfig())


def test_build_cannot_run_twice():
    node = HarnessPipeline().create(AutoSupportPlane)
    node.build(*build_args())

    with pytest.raises(RuntimeError):
        node.build(*build_args())


class Selector:
    """A ``selectCandidate`` that records each call and answers with ``choose``."""

    def __init__(self, choose):
        self.choose = choose
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.choose(*args, **kwargs)


def farthest(candidates):
    return max(candidates, key=lambda c: abs(c.plane.offset_mm))


def selecting(selector):
    """An ``ON_START`` harness whose node is built with ``selectCandidate``."""
    return Harness(on_start(), selectCandidate=selector)


def test_custom_selector_gets_the_whole_tuple_and_its_pick_is_sent():
    selector = Selector(farthest)
    node = selecting(selector)

    result = first_fit(node, scripted_floors((-1000.0, 200), (-1500.0, 100)))

    assert result.passed is True
    assert result.plane == floor(-1500)
    ((args, kwargs),) = selector.calls
    assert kwargs == {}
    (candidates,) = args
    assert isinstance(candidates, tuple)
    assert {c.plane for c in candidates} == {floor(-1000), floor(-1500)}


def test_selector_none_keeps_searching_and_the_next_fit_asks_again():
    selector = Selector(
        lambda candidates: None if len(selector.calls) < 3 else farthest(candidates)
    )
    node = selecting(selector)
    scene = scripted_floors((-1000.0, 200), (-1500.0, 100))
    node.gravity()
    with scripted(scene):
        node.plane_cloud()
        node.run(2)
        for _ in range(2):
            assert node.no_result()
            node.gravity()
            node.plane_cloud()
            node.run()

    assert node.result().plane == floor(-1500)
    assert len(selector.calls) == 3


class SelectorError(Exception):
    pass


def raising(candidates):
    raise SelectorError


def stranger(candidates):
    """Return a candidate equal to none of ``candidates``."""
    return SupportPlaneCandidate(plane=floor(-1234), inlier_count=1, inlier_ratio=1.0)


@pytest.mark.parametrize(
    ("selector", "error"), [(raising, SelectorError), (stranger, ValueError)]
)
def test_a_failing_selector_raises_out_of_a_run_pass(selector, error):
    node = selecting(selector)
    node.gravity()
    with scripted(scripted_floors((-1000.0, 200))):
        node.plane_cloud()
        node.run()

        with pytest.raises(error):
            node.run()


def test_build_rejects_a_selector_that_is_not_callable():
    node = HarnessPipeline().create(AutoSupportPlane)

    with pytest.raises(TypeError):
        node.build(*build_args(), selectCandidate=floor(-1000))


# Big enough for the calibrator to measure an on-plane ratio (625 sampled points).
LARGE_CLOUD_SIZE = (100, 100)
# Enough inliers to fit a plane in a large cloud.
LARGE_INLIER_COUNT = 9000
INTERVAL = 2.0


def continuous(failed_checks_before_refit=2, **calibration):
    calibration = {"max_plane_passes": 3, "min_on_plane_ratio": 0.5, **calibration}
    return AutoSupportPlaneConfig(
        mode=AutoSupportPlaneConfig.Mode.CONTINUOUS,
        check_interval_sec=INTERVAL,
        failed_checks_before_refit=failed_checks_before_refit,
        calibration=PlaneCalibrationConfig(**calibration),
    )


def holding(config, **build_kwargs):
    """A harness on large clouds whose first fit holds the floor 1000 mm away."""
    node = Harness(config, LARGE_CLOUD_SIZE, **build_kwargs)
    node.gravity()
    with scripted(scripted_floors((-1000.0, LARGE_INLIER_COUNT))):
        node.plane_cloud()
        node.run(2)
    assert node.result().plane == floor(-1000)
    return node


def check(node, clock, depth=1000.0, wait=INTERVAL, **gravity):
    """Let ``wait`` seconds pass, then send fresh IMU and a cloud, and run a check.

    ``depth`` is one depth for the whole cloud in mm, or an array of depths.

    Returns:
        The sent cloud.
    """
    clock.now += wait
    node.gravity(**gravity)
    width, height = node.size
    cloud = node.depth_cloud(np.broadcast_to(depth, (height, width)))
    node.run(2)
    return cloud


def test_passing_check_sends_the_plane_every_interval(clock):
    node = holding(continuous())

    for _ in range(2):
        check(node, clock, wait=INTERVAL - 0.01)
        assert node.no_result()
        cloud = check(node, clock, wait=0.01)

        result = node.result()
        assert result.passed is True
        assert result.plane == floor(-1000)
        assert_stamped_from(result, cloud)


# A cloud hiding the held floor 1000 mm away behind something 600 mm away.
OCCLUDED = 600.0


def refit_cloud(node, depth=1000.0, **gravity):
    """Send fresh IMU and a cloud and run one pass, fitting it in a running refit.

    Returns:
        The sent cloud.
    """
    node.gravity(**gravity)
    cloud = node.plane_cloud(depth)
    node.run()
    return cloud


def test_bumped_camera_refits_to_the_new_plane_and_checks_it(clock):
    node = holding(continuous(failed_checks_before_refit=2))
    check(node, clock, depth=1300)
    check(node, clock, depth=1300)
    assert node.no_result()

    with scripted(scripted_floors((-1300.0, LARGE_INLIER_COUNT))):
        cloud = refit_cloud(node, depth=1300)

    result = node.result()
    assert result.passed is True
    assert result.plane == floor(-1300)
    assert_stamped_from(result, cloud)
    check(node, clock, depth=1300, wait=INTERVAL - 0.01)
    assert node.no_result()
    check(node, clock, depth=1300, wait=0.01)
    assert node.result().plane == floor(-1300)


def test_failed_refit_keeps_the_old_plane_and_checks_it_again(clock):
    node = holding(continuous(failed_checks_before_refit=2, max_calibration_attempts=1))
    check(node, clock, depth=OCCLUDED)
    check(node, clock, depth=OCCLUDED)

    clock.now += INTERVAL
    cloud = refit_cloud(node, depth=0)

    result = node.result()
    assert result.passed is False
    assert result.plane == floor(-1000)
    assert_stamped_from(result, cloud)
    check(node, clock, wait=INTERVAL - 0.01)
    assert node.no_result()
    check(node, clock, wait=0.01)
    assert node.result().passed is True


def test_occlusion_keeps_the_plane_through_a_refit_that_finds_nothing(clock):
    node = holding(continuous(failed_checks_before_refit=2))
    with scripted(ScriptedScene([])):
        for _ in range(2):
            check(node, clock, depth=OCCLUDED)
            check(node, clock)
            assert node.result().plane == floor(-1000)
        check(node, clock, depth=OCCLUDED)
        check(node, clock, depth=OCCLUDED)

        for _ in range(5):
            check(node, clock)
        assert node.no_result()


def test_one_check_is_in_flight_through_an_imu_stall(clock):
    node = holding(continuous())
    for _ in range(10):
        clock.now += INTERVAL
        node.plane_cloud()
        node.run(2)
    assert node.no_result()

    check(node, clock, wait=0)

    assert node.result().plane == floor(-1000)
    assert node.no_result()


def sparse_depths():
    """Depths on the held floor, with too few points to measure an on-plane ratio."""
    width, height = LARGE_CLOUD_SIZE
    depths = np.full((height, width), 1000.0)
    depths[:40] = 0
    return depths


@pytest.mark.parametrize(
    ("min_on_plane_ratio", "depth", "passes"),
    [(0.5, OCCLUDED, False), (0.0, OCCLUDED, True), (0.0, sparse_depths(), False)],
    ids=["off-plane", "off-plane-any-ratio", "sparse"],
)
def test_check_passes_on_a_measured_on_plane_ratio(
    clock, min_on_plane_ratio, depth, passes
):
    node = holding(continuous(min_on_plane_ratio=min_on_plane_ratio))

    check(node, clock, depth=depth)

    assert node.no_result() is not passes


def test_camera_tipped_past_the_level_limit_fails_checks_and_refits(clock):
    config = continuous(
        failed_checks_before_refit=2,
        max_calibration_attempts=1,
        max_gravity_angle_deg=5,
    )
    node = holding(config)
    tipped = np.deg2rad(8)
    gravity = {"y": -9.8 * np.sin(tipped), "z": -9.8 * np.cos(tipped)}
    check(node, clock, **gravity)
    check(node, clock, **gravity)
    assert node.no_result()

    refit_cloud(node, **gravity)

    result = node.result()
    assert result.passed is False
    assert result.plane == floor(-1000)


def test_selector_chooses_on_every_refit(clock):
    selector = Selector(farthest)
    node = holding(continuous(failed_checks_before_refit=1), selectCandidate=selector)
    check(node, clock, depth=1300)

    scene = scripted_floors((-1300.0, LARGE_INLIER_COUNT), (-1600.0, 500))
    with scripted(scene):
        refit_cloud(node, depth=1300)

    assert node.result().plane == floor(-1600)
    assert len(selector.calls) == 2


def test_continuous_failed_attempt_without_a_plane_is_retried():
    node = Harness(continuous(max_calibration_attempts=1), LARGE_CLOUD_SIZE)
    node.run()
    fit_empty_cloud(node)
    assert node.result().plane is None

    with scripted(scripted_floors((-1000.0, LARGE_INLIER_COUNT))):
        refit_cloud(node)

    assert node.result().plane == floor(-1000)


def test_no_plane_checks_in_on_start(clock):
    node = holding(on_start())

    for _ in range(3):
        check(node, clock)

    assert node.no_result()
