import depthai as dai
import pytest

from depthai_nodes.node import HostParsingNeuralNetwork
from tests.utils import InputMock, NeuralNetworkMock, PipelineMock
from tests.utils.nodes.mocks.pipeline import DetectionParserMock


def get_model_archive(model_name: str) -> dai.NNArchive:
    try:
        return dai.NNArchive(
            dai.getModelFromZoo(dai.NNModelDescription(model_name, "RVC2"))
        )
    except RuntimeError as exc:
        if "No internet connection available" in str(exc):
            pytest.skip(f"Model zoo unavailable for {model_name}: {exc}")
        raise


@pytest.fixture
def pipeline():
    dai.node.NeuralNetwork = NeuralNetworkMock
    return PipelineMock()


def test_yolo(pipeline: PipelineMock):
    nn_archive = get_model_archive(
        "luxonis/yolov8-instance-segmentation-nano:coco-512x288"
    )
    nn = pipeline.create(HostParsingNeuralNetwork).build(
        input=InputMock(), nnSource=nn_archive, fps=30
    )
    parser = nn.getParser()
    assert isinstance(parser, DetectionParserMock)


def test_unsupported(pipeline: PipelineMock):
    nn_archive = get_model_archive("luxonis/mobilenet-ssd:300x300")
    nn = pipeline.create(HostParsingNeuralNetwork)
    with pytest.raises(ValueError):
        nn.build(input=InputMock(), nnSource=nn_archive, fps=30)
