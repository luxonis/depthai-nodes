import depthai as dai
import pytest
from stability_tests.conftest import Input, NeuralNetworkMock, PipelineMock

from depthai_nodes.node import HostParsingNeuralNetwork, YOLOExtendedParser


@pytest.fixture
def pipeline():
    dai.node.NeuralNetwork = NeuralNetworkMock
    return PipelineMock()


def test_initialization(pipeline: PipelineMock):
    assert pipeline.create(HostParsingNeuralNetwork).parentInitialized()


def test_yolo(pipeline: PipelineMock):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription("luxonis/yolov6-nano", "RVC2"))
    )
    nn = pipeline.create(HostParsingNeuralNetwork).build(
        input=Input(), nn_source=nn_archive, fps=30
    )
    parser = nn.getParser()
    assert isinstance(parser, YOLOExtendedParser)


def test_unsupported(pipeline: PipelineMock):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription("luxonis/mobilenet-ssd:300x300", "RVC2")
        )
    )
    nn = pipeline.create(HostParsingNeuralNetwork)
    with pytest.raises(ValueError):
        nn.build(input=Input(), nn_source=nn_archive, fps=30)
