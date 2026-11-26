from typing import Any, cast

import depthai as dai
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import SupportPlane  # noqa: E402
from depthai_nodes.node import (  # noqa: E402
    BoxDetector,
    BoxDimensioner,
    BoxMeasurer,
)
from depthai_nodes.node.box_associator.box_associator import MAX_VIEWS  # noqa: E402
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler  # noqa: E402
from depthai_nodes.node.box_detector.box_detection_assembler import (  # noqa: E402
    BoxDetectionAssembler,
)
from tests.utils import (  # noqa: E402
    DetectionNetworkMock,
    DeviceMock,
    ImageManipMock,
    ObjectTrackerMock,
    OutputMock,
    PipelineMock,
    PointCloudMock,
)
from tests.utils.nodes.mocks.pipeline import DetectionParserMock  # noqa: E402

FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


class FakeArchive:
    """Offline stand-in for the box model's archive: a 512x320 input."""

    def __init__(self, path):
        pass

    def getInputSize(self):
        return (512, 320)


def rvc4(deviceId: str) -> DeviceMock:
    device = DeviceMock(deviceId)
    device.setPlatform(dai.Platform.RVC4)
    return device


A, B = rvc4("A"), rvc4("B")


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineMock:
    monkeypatch.setattr(dai.node, "PointCloud", PointCloudMock)
    monkeypatch.setattr(dai.node, "DetectionNetwork", DetectionNetworkMock)
    monkeypatch.setattr(dai.node, "ImageManip", ImageManipMock)
    monkeypatch.setattr(dai.node, "ObjectTracker", ObjectTrackerMock)
    monkeypatch.setattr(dai, "getModelFromZoo", lambda description: "box.tar.xz")
    monkeypatch.setattr(dai, "NNArchive", FakeArchive)
    return PipelineMock()


def view(device: DeviceMock) -> tuple[OutputMock, OutputMock]:
    """Return a ``(color on device, depth)`` view."""
    return OutputMock(device), OutputMock()


def create(pipeline: PipelineMock) -> BoxMeasurer:
    return cast(BoxMeasurer, pipeline.create(cast(Any, BoxMeasurer)))


def build(pipeline: PipelineMock, views=None, **kwargs) -> BoxMeasurer:
    """Build a measurer on ``views``, by default one view on device A."""
    return create(pipeline).build(cast(Any, views or [view(A)]), **kwargs)


def find_nodes(pipeline: PipelineMock, node_type: type) -> list:
    return [n for n in pipeline.getAllNodes() if isinstance(n, node_type)]


def find_node(pipeline: PipelineMock, node_type: type):
    (node,) = find_nodes(pipeline, node_type)
    return node


def detector_device(pipeline: PipelineMock, detections) -> DeviceMock:
    """Return the device of the detector whose output is ``detections``."""
    (detector,) = [d for d in find_nodes(pipeline, BoxDetector) if d.out is detections]
    # The pipeline mock shadows ``out``; read the detector's own property.
    assembler_out = BoxDetector.out.fget(detector)  # pyright: ignore[reportOptionalCall]
    (assembler,) = [
        a for a in find_nodes(pipeline, BoxDetectionAssembler) if a.out is assembler_out
    ]
    (tracker,) = [
        t
        for t in find_nodes(pipeline, ObjectTrackerMock)
        if t.out is assembler._linked_args[0]
    ]
    return tracker.device


def parsers(pipeline: PipelineMock) -> list[DetectionParserMock]:
    return [nn.detectionParser for nn in find_nodes(pipeline, DetectionNetworkMock)]


def sent_thresholds(parser: DetectionParserMock) -> list[float]:
    thresholds, config_input = [], parser.inputConfig
    while (config := config_input.tryGet()) is not None:
        thresholds.append(round(config.getConfidenceThreshold(), 2))
    return thresholds


def test_each_view_gets_a_detector_on_its_own_device(pipeline: PipelineMock):
    devices = [B, A]

    build(pipeline, [view(device) for device in devices])

    assert len(find_nodes(pipeline, BoxDetector)) == 2
    nn_devices = [nn.device for nn in find_nodes(pipeline, DetectionNetworkMock)]
    assert [d.getDeviceId() for d in nn_devices] == ["B", "A"]


def test_view_order_is_kept_through_the_associator(pipeline: PipelineMock):
    devices = [B, A]
    views = [view(device) for device in devices]

    build(pipeline, views)

    cloud = find_node(pipeline, PointCloudMock)
    labelers = find_nodes(pipeline, BoxLabeler)
    for i, (device, (_, depth)) in enumerate(zip(devices, views)):
        assert cloud.getDepthInput(str(i)) in depth.links
        (labeler,) = [n for n in labelers if n._linked_args[1] is depth]
        assert detector_device(pipeline, labeler._linked_args[0]) is device


def test_build_threshold_reaches_every_parser(pipeline: PipelineMock):
    build(pipeline, [view(A), view(B)], confidenceThreshold=0.65)

    thresholds = [p.getConfidenceThreshold() for p in parsers(pipeline)]
    assert thresholds == pytest.approx([0.65, 0.65])


def test_live_threshold_reaches_every_parser(pipeline: PipelineMock):
    measurer = build(pipeline, [view(A), view(B)])

    measurer.setConfidenceThreshold(0.7)

    assert [sent_thresholds(p) for p in parsers(pipeline)] == [[0.7], [0.7]]


def test_threshold_cannot_be_set_before_build(pipeline: PipelineMock):
    measurer = create(pipeline)
    with pytest.raises(RuntimeError):
        measurer.setConfidenceThreshold(0.7)


def test_plane_setters_need_no_build(pipeline: PipelineMock):
    measurer = create(pipeline)

    measurer.setSupportPlane(FLOOR)
    assert measurer.getSupportPlane() == FLOOR
    measurer.clearSupportPlane()
    assert measurer.getSupportPlane() is None


def test_build_sets_the_support_plane(pipeline: PipelineMock):
    measurer = build(pipeline, supportPlane=FLOOR)

    assert measurer.getSupportPlane() == FLOOR


def test_out_is_the_dimensioner_out(pipeline: PipelineMock):
    measurer = build(pipeline)
    dimensioner = find_node(pipeline, BoxDimensioner)

    # The pipeline mock's node class defines its own ``out``; read BoxMeasurer's.
    assert BoxMeasurer.out.fget(measurer) is dimensioner.out  # pyright: ignore[reportOptionalCall]


@pytest.mark.parametrize(
    ("views", "kwargs", "error"),
    [
        (None, {"confidenceThreshold": 1.1}, ValueError),
        (None, {"confidenceThreshold": "0.5"}, TypeError),
        (None, {"supportPlane": "floor"}, TypeError),
        ([], {}, ValueError),
        ([view(A) for _ in range(MAX_VIEWS + 1)], {}, ValueError),
    ],
)
def test_invalid_build_arguments_are_rejected_before_the_model_is_fetched(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch, views, kwargs, error
):
    def no_fetch(description):
        raise AssertionError("the model was fetched")

    monkeypatch.setattr(dai, "getModelFromZoo", no_fetch)
    measurer = create(pipeline)

    with pytest.raises(error):
        measurer.build(cast(Any, [view(A)] if views is None else views), **kwargs)


def offline(description):
    raise ConnectionError("zoo unreachable")


@pytest.mark.parametrize("first_build_fails", [False, True])
def test_a_second_build_is_rejected(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch, first_build_fails
):
    measurer = create(pipeline)
    if first_build_fails:
        with monkeypatch.context() as patch:
            patch.setattr(dai, "getModelFromZoo", offline)
            with pytest.raises(ConnectionError):
                measurer.build(cast(Any, [view(A)]))
    else:
        measurer.build(cast(Any, [view(A)]))

    with pytest.raises(RuntimeError):
        measurer.build(cast(Any, [view(A)]))
