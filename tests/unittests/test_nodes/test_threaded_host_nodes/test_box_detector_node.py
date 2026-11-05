from types import SimpleNamespace

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.node import BoxDetector
from depthai_nodes.node.box_detector.box_detection_assembler import (
    BoxDetectionAssembler,
)
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


def emitted_confidences(
    pipeline: PipelineMock, confidences: list[float]
) -> list[float]:
    """Feed side-by-side tracked boxes and return the confidences emitted."""
    tracklets, items = [], []
    mask = np.full((2, 2 * len(confidences)), 255, dtype=np.uint8)
    for index, confidence in enumerate(confidences):
        xmin, xmax = index / len(confidences), (index + 1) / len(confidences)
        detection = dai.ImgDetection()
        detection.confidence = confidence
        detection.xmin, detection.ymin, detection.xmax, detection.ymax = (
            xmin,
            0,
            xmax,
            1,
        )
        items.append(detection)
        tracklet = dai.Tracklet()
        tracklet.id = index
        tracklet.status = dai.Tracklet.TrackingStatus.TRACKED
        tracklet.roi = dai.Rect(dai.Point2f(xmin, 0), dai.Point2f(xmax, 1))
        tracklets.append(tracklet)
        mask[:, 2 * index : 2 * index + 2] = index
    detections = dai.ImgDetections()
    detections.detections = items
    detections.setCvSegmentationMask(mask)
    tracks = dai.Tracklets()
    tracks.tracklets = tracklets
    assembler = find_node(pipeline, BoxDetectionAssembler)
    queue = assembler.out.createOutputQueue()
    assembler.process(tracks, detections)
    return [round(item.confidence, 2) for item in queue.get().items]


def test_build_fixes_the_parser_threshold_at_zero(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=0.65)

    assert find_node(pipeline, DetectionParserMock).getConfidenceThreshold() == 0


def test_build_threshold_drops_detections_below_it(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=0.65)

    assert emitted_confidences(pipeline, [0.6, 0.7]) == [0.7]


def test_default_threshold_is_one_half(pipeline: PipelineMock):
    pipeline.create(BoxDetector).build(OutputMock())

    assert emitted_confidences(pipeline, [0.4, 0.5]) == [0.5]


def test_live_threshold_changes_the_emitted_detections(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(OutputMock())

    detector.setConfidenceThreshold(0.7)

    assert emitted_confidences(pipeline, [0.6, 0.8]) == [0.8]
    assert find_node(pipeline, DetectionParserMock).getConfidenceThreshold() == 0


def test_lowering_the_live_threshold_emits_detections_again(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(OutputMock(), confidenceThreshold=0.7)
    assert emitted_confidences(pipeline, [0.4, 0.8]) == [0.8]

    detector.setConfidenceThreshold(0.3)

    assert emitted_confidences(pipeline, [0.4, 0.8]) == [0.4, 0.8]


def test_threshold_accepts_bounds_and_numpy_scalars(pipeline: PipelineMock):
    detector = pipeline.create(BoxDetector).build(
        OutputMock(), confidenceThreshold=np.float32(0.25)
    )
    assert emitted_confidences(pipeline, [0.2, 0.3]) == [0.3]

    detector.setConfidenceThreshold(0)
    detector.setConfidenceThreshold(1)
    assert emitted_confidences(pipeline, [0.9, 1.0]) == [1.0]


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
