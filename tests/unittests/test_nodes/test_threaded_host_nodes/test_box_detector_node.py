import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import BoxDetector
from depthai_nodes.node.box_detector.box_detection_assembler import (
    BoxDetectionAssembler,
)
from tests.utils import (
    DetectionNetworkMock,
    DeviceMock,
    ImageManipMock,
    ObjectTrackerMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.pipeline import DetectionParserMock

INVALID_VALUES = [float("nan"), float("inf"), -0.1, 1.1]
INVALID_TYPES = ["0.5", None, True]


class FakeArchive:
    """Offline stand-in for the box model's archive: a 512x320 input."""

    def __init__(self, path):
        pass

    def getInputSize(self):
        return (512, 320)


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineMock:
    monkeypatch.setattr(dai.node, "DetectionNetwork", DetectionNetworkMock)
    monkeypatch.setattr(dai.node, "ImageManip", ImageManipMock)
    monkeypatch.setattr(dai.node, "ObjectTracker", ObjectTrackerMock)
    monkeypatch.setattr(dai, "getModelFromZoo", lambda description: "box.tar.xz")
    monkeypatch.setattr(dai, "NNArchive", FakeArchive)
    pipeline = PipelineMock()
    pipeline.getDefaultDevice().setPlatform(dai.Platform.RVC4)
    return pipeline


def find_node(pipeline: PipelineMock, node_type: type):
    (node,) = [n for n in pipeline.getAllNodes() if isinstance(n, node_type)]
    return node


def parser(pipeline: PipelineMock) -> DetectionParserMock:
    return find_node(pipeline, DetectionNetworkMock).detectionParser


def sent_configs(pipeline: PipelineMock) -> list[dai.DetectionParserConfig]:
    configs, config_input = [], parser(pipeline).inputConfig
    while (config := config_input.tryGet()) is not None:
        configs.append(config)
    return configs


def test_build_links_color_through_the_model_into_the_tracker(
    pipeline: PipelineMock,
):
    color = OutputMock()
    pipeline.create(BoxDetector).build(color)

    resize = find_node(pipeline, ImageManipMock)
    tracker = find_node(pipeline, ObjectTrackerMock)
    nn = find_node(pipeline, DetectionNetworkMock)
    assert any(link is resize.inputImage for link in color.links)
    assert any(link is nn.input for link in resize.out.links)
    assert any(link is tracker.inputTrackerFrame for link in color.links)
    assert nn.out.links == [tracker.inputDetections]
    assembler = find_node(pipeline, BoxDetectionAssembler)
    assert assembler._linked_args == (tracker.out, tracker.passthroughDetections)


def subnode_devices(pipeline: PipelineMock) -> list:
    return [
        find_node(pipeline, node_type).device
        for node_type in (ImageManipMock, DetectionNetworkMock, ObjectTrackerMock)
    ]


def test_subnodes_run_on_the_color_streams_device(pipeline: PipelineMock):
    device = DeviceMock("B")
    device.setPlatform(dai.Platform.RVC4)
    pipeline.create(BoxDetector).build(OutputMock(device))

    assert subnode_devices(pipeline) == [device, device, device]


def test_a_host_color_stream_uses_the_default_device(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock())

    default = pipeline.getDefaultDevice()
    assert subnode_devices(pipeline) == [default, default, default]


def fetch_recorder(monkeypatch: pytest.MonkeyPatch) -> list:
    fetched = []
    monkeypatch.setattr(
        dai,
        "getModelFromZoo",
        lambda description: fetched.append(description) or "box.tar.xz",
    )
    return fetched


def test_an_rvc4_color_device_builds_with_an_rvc2_default(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch
):
    fetched = fetch_recorder(monkeypatch)
    pipeline.getDefaultDevice().setPlatform(dai.Platform.RVC2)
    device = DeviceMock("B")
    device.setPlatform(dai.Platform.RVC4)

    pipeline.create(BoxDetector).build(OutputMock(device))

    assert [d.platform for d in fetched] == ["RVC4"]


def test_an_rvc2_color_device_is_rejected_with_an_rvc4_default(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch
):
    fetched = fetch_recorder(monkeypatch)
    pipeline.getDefaultDevice().setPlatform(dai.Platform.RVC4)
    device = DeviceMock("B")
    device.setPlatform(dai.Platform.RVC2)
    detector = pipeline.create(BoxDetector)

    with pytest.raises(ValueError):
        detector.build(OutputMock(device))
    assert fetched == []


def test_build_sets_the_parser_threshold(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=0.65)

    assert parser(pipeline).getConfidenceThreshold() == pytest.approx(0.65)
    assert sent_configs(pipeline) == []


def test_default_threshold_is_one_half(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock())

    assert parser(pipeline).getConfidenceThreshold() == 0.5


def test_live_threshold_sends_a_parser_config(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(OutputMock())
    parser(pipeline).setIouThreshold(0.4)

    detector.setConfidenceThreshold(0.7)
    detector.setConfidenceThreshold(0.3)

    configs = sent_configs(pipeline)
    assert [round(c.getConfidenceThreshold(), 2) for c in configs] == [0.7, 0.3]
    assert [round(c.getIouThreshold(), 2) for c in configs] == [0.4, 0.4]


def test_threshold_accepts_bounds_and_numpy_scalars(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(
        OutputMock(), confidenceThreshold=np.float32(0.25)
    )
    assert parser(pipeline).getConfidenceThreshold() == pytest.approx(0.25)

    detector.setConfidenceThreshold(0)
    detector.setConfidenceThreshold(np.float64(1))
    configs = sent_configs(pipeline)
    assert [c.getConfidenceThreshold() for c in configs] == [0, 1]


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_invalid_threshold_values_are_rejected(pipeline: PipelineMock, value):
    detector = pipeline.create(BoxDetector).build(OutputMock())
    with pytest.raises(ValueError):
        detector.setConfidenceThreshold(value)
    with pytest.raises(ValueError):
        pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=value)


@pytest.mark.parametrize("value", INVALID_TYPES)
def test_non_numeric_thresholds_are_type_errors(pipeline: PipelineMock, value):
    detector = pipeline.create(BoxDetector).build(OutputMock())
    with pytest.raises(TypeError):
        detector.setConfidenceThreshold(value)
    with pytest.raises(TypeError):
        pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=value)


def test_threshold_cannot_be_set_before_build(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector)
    with pytest.raises(RuntimeError):
        detector.setConfidenceThreshold(0.7)


def test_a_second_build_is_rejected(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(OutputMock())
    with pytest.raises(RuntimeError):
        detector.build(OutputMock())
