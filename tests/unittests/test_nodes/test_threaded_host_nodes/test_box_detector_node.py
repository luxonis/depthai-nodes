from types import SimpleNamespace

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import BoxDetector
from tests.utils import (
    ImageManipMock,
    NeuralNetworkMock,
    ObjectTrackerMock,
    OutputMock,
    PipelineMock,
)
from tests.utils.nodes.mocks.pipeline import DetectionParserMock

INVALID_VALUES = [float("nan"), float("inf"), -0.1, 1.1]
INVALID_TYPES = ["0.5", None, True]


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


def find_node(pipeline: PipelineMock, node_type: type):
    (node,) = [n for n in pipeline.getAllNodes() if isinstance(n, node_type)]
    return node


def parser(pipeline: PipelineMock) -> DetectionParserMock:
    return find_node(pipeline, DetectionParserMock)


def sent_configs(pipeline: PipelineMock) -> list[dai.DetectionParserConfig]:
    configs, config_input = [], parser(pipeline).inputConfig
    while (config := config_input.tryGet()) is not None:
        configs.append(config)
    return configs


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


@pytest.mark.parametrize("missing", ["inputConfig", "DetectionParserConfig"])
def test_live_threshold_needs_depthai_3_11(
    pipeline: PipelineMock, monkeypatch: pytest.MonkeyPatch, missing: str
):
    owner = DetectionParserMock if missing == "inputConfig" else dai
    monkeypatch.delattr(owner, missing)
    detector = pipeline.create(BoxDetector).build(
        OutputMock(), confidenceThreshold=0.65
    )

    assert parser(pipeline).getConfidenceThreshold() == pytest.approx(0.65)
    with pytest.raises(RuntimeError, match="3.11"):
        detector.setConfidenceThreshold(0.7)


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
