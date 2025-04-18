import depthai as dai
import pytest

from depthai_nodes.node import ParsingNeuralNetwork
from tests.utils import InputMock, NeuralNetworkMock, PipelineMock


@pytest.fixture
def pipeline():
    dai.node.NeuralNetwork = NeuralNetworkMock
    return PipelineMock()


def validate(pnn: ParsingNeuralNetwork, nn_archive: dai.NNArchive):
    num_heads = len(nn_archive.getConfig().model.heads)

    assert (
        len(pnn._parsers) == num_heads
    ), f"Expected {num_heads} parsers, got {len(pnn._parsers)}"

    for parser_ix in range(num_heads):
        p = pnn.getParser(parser_ix)
        assert p is not None, f"Parser {parser_ix} is None"

        output = pnn.getOutput(parser_ix)
        assert output is not None, f"Output {parser_ix} is None"


@pytest.mark.parametrize(
    "model",
    [
        "luxonis/yunet:320x240",
        "luxonis/vehicle-attributes-classification:72x72",
        "luxonis/mediapipe-hand-landmarker:224x224",
        "luxonis/yolov6-nano:r2-coco-512x288",
        "luxonis/mobilenet-ssd:300x300",
    ],
)
def test_parsing_neural_network(pipeline: PipelineMock, model: str):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription(model, "RVC2"))
    )
    pnn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input=InputMock(), nn_source=nn_archive, fps=30
    )

    validate(pnn, nn_archive)


def test_parsing_neural_network_nn_archive(pipeline: PipelineMock):
    model = "luxonis/yunet:320x240"
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription(model, "RVC2"))
    )
    pnn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input=InputMock(), nn_source=nn_archive, fps=30
    )

    validate(pnn, nn_archive)

    model = "luxonis/vehicle-attributes-classification:72x72"
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription(model, "RVC2"))
    )
    pnn.setNNArchive(nn_archive)

    validate(pnn, nn_archive)
