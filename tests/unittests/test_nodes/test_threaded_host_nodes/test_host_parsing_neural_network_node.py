import depthai as dai
import pytest

from depthai_nodes.node import HostParsingNeuralNetwork
from tests.utils.nodes.mocks.pipeline import DetectionParserMock
from tests.utils import InputMock, NeuralNetworkMock, PipelineMock


@pytest.fixture
def pipeline():
    dai.node.NeuralNetwork = NeuralNetworkMock
    return PipelineMock()


def test_yolo(pipeline: PipelineMock):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription("luxonis/yolov8-instance-segmentation-nano:coco-512x288", "RVC2"))
    )
    nn = pipeline.create(HostParsingNeuralNetwork).build(
        input=InputMock(), nn_source=nn_archive, fps=30
    )
    parser = nn.getParser()
    assert isinstance(parser, DetectionParserMock)


def test_unsupported(pipeline: PipelineMock):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription("luxonis/mobilenet-ssd:300x300", "RVC2")
        )
    )
    nn = pipeline.create(HostParsingNeuralNetwork)
    with pytest.raises(ValueError):
        nn.build(input=InputMock(), nn_source=nn_archive, fps=30)
