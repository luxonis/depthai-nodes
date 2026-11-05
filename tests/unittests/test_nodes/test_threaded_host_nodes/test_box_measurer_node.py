from types import SimpleNamespace
from typing import Any, cast

import depthai as dai
import pytest

pytest.importorskip("open3d")

from depthai_nodes.message import SupportPlane  # noqa: E402
from depthai_nodes.node import BoxDimensioner, BoxMeasurer  # noqa: E402
from tests.utils import (  # noqa: E402
    ImageManipMock,
    NeuralNetworkMock,
    ObjectTrackerMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.pipeline import DetectionParserMock  # noqa: E402

CLOUD_SIZE = (64, 48)
FLOOR = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


class FakeArchive:
    """Offline stand-in for the box model's archive: one YOLO head, 512x320."""

    def __init__(self, path):
        head = SimpleNamespace(parser="YOLO", metadata=SimpleNamespace())
        self._config = SimpleNamespace(model=SimpleNamespace(heads=[head], inputs=[]))

    def getConfig(self):
        return self._config

    def getInputSize(self):
        return (512, 320)


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineMock:
    monkeypatch.setattr(dai.node, "NeuralNetwork", NeuralNetworkMock)
    monkeypatch.setattr(dai.node, "ImageManip", ImageManipMock)
    monkeypatch.setattr(dai.node, "ObjectTracker", ObjectTrackerMock)
    monkeypatch.setattr(dai, "getModelFromZoo", lambda description: "box.tar.xz")
    monkeypatch.setattr(dai, "NNArchive", FakeArchive)
    pipeline = PipelineMock()
    pipeline.getDefaultDevice().setPlatform(dai.Platform.RVC4)
    return pipeline


def output() -> dai.Node.Output:
    return cast(dai.Node.Output, OutputMock())


def create(pipeline: PipelineMock) -> BoxMeasurer:
    return cast(BoxMeasurer, pipeline.create(cast(Any, BoxMeasurer)))


def build(pipeline: PipelineMock, **kwargs) -> BoxMeasurer:
    return create(pipeline).build(output(), output(), CLOUD_SIZE, **kwargs)


def find_node(pipeline: PipelineMock, node_type: type):
    (node,) = [n for n in pipeline.getAllNodes() if isinstance(n, node_type)]
    return node


def parser(pipeline: PipelineMock) -> DetectionParserMock:
    return find_node(pipeline, DetectionParserMock)


def sent_thresholds(pipeline: PipelineMock) -> list[float]:
    thresholds, config_input = [], parser(pipeline).inputConfig
    while (config := config_input.tryGet()) is not None:
        thresholds.append(round(config.getConfidenceThreshold(), 2))
    return thresholds


def test_build_sets_the_detection_threshold(pipeline: PipelineMock):
    build(pipeline, confidenceThreshold=0.65)

    assert parser(pipeline).getConfidenceThreshold() == pytest.approx(0.65)


def test_default_threshold_is_one_half(pipeline: PipelineMock):
    build(pipeline)

    assert parser(pipeline).getConfidenceThreshold() == 0.5


def test_live_threshold_reaches_the_parser(pipeline: PipelineMock):
    measurer = build(pipeline)

    measurer.setConfidenceThreshold(0.7)

    assert sent_thresholds(pipeline) == [0.7]


def test_live_threshold_needs_depthai_3_11(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delattr(dai, "DetectionParserConfig")
    measurer = build(pipeline, confidenceThreshold=0.65)

    assert parser(pipeline).getConfidenceThreshold() == pytest.approx(0.65)
    with pytest.raises(RuntimeError, match="3.11"):
        measurer.setConfidenceThreshold(0.7)


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


def test_outputs_are_the_dimensioner_outputs(pipeline: PipelineMock):
    measurer = build(pipeline)
    dimensioner = find_node(pipeline, BoxDimensioner)

    # The pipeline mock's node class defines its own ``out``; read BoxMeasurer's.
    assert BoxMeasurer.out.fget(measurer) is dimensioner.out  # pyright: ignore[reportOptionalCall]
    assert measurer.boxCloudOutput is dimensioner.boxCloudOutput


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidenceThreshold": 1.1}, ValueError),
        ({"confidenceThreshold": "0.5"}, TypeError),
        ({"supportPlane": "floor"}, TypeError),
    ],
)
def test_invalid_build_arguments_are_rejected(pipeline: PipelineMock, kwargs, error):
    with pytest.raises(error):
        build(pipeline, **kwargs)


@pytest.mark.parametrize(
    ("size", "error"), [((0, 48), ValueError), ((64.0, 48), TypeError)]
)
def test_invalid_cloud_size_is_rejected_before_the_model_is_fetched(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch, size, error
):
    def no_fetch(description):
        raise AssertionError("model fetched")

    monkeypatch.setattr(dai, "getModelFromZoo", no_fetch)
    with pytest.raises(error):
        create(pipeline).build(output(), output(), size)


def test_a_second_build_is_rejected(pipeline: PipelineMock):
    measurer = build(pipeline)
    with pytest.raises(RuntimeError):
        measurer.build(output(), output(), CLOUD_SIZE)


def test_build_can_be_retried_after_the_detector_fails(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch
):
    def offline(description):
        raise ConnectionError("zoo unreachable")

    measurer = create(pipeline)
    with monkeypatch.context() as patch:
        patch.setattr(dai, "getModelFromZoo", offline)
        with pytest.raises(ConnectionError):
            measurer.build(output(), output(), CLOUD_SIZE)

    measurer.build(output(), output(), CLOUD_SIZE)
