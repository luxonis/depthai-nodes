import depthai as dai
import pytest

from depthai_nodes.node import ParserGenerator, YOLOExtendedParser
from tests.utils import PipelineMock


@pytest.fixture
def parser_generator():
    pipeline = PipelineMock()
    return pipeline.create(ParserGenerator)


@pytest.mark.parametrize(
    "model_name",
    [
        "luxonis/yunet:320x240",
        "luxonis/vehicle-attributes-classification:72x72",
        "luxonis/mediapipe-hand-landmarker:224x224",
        "luxonis/yolov6-nano:r2-coco-512x288",
        "luxonis/mobilenet-ssd:300x300",
    ],
)
def test_parser_generator(parser_generator: ParserGenerator, model_name: str):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription(model_name, "RVC2"))
    )

    num_heads = len(nn_archive.getConfig().model.heads)

    parsers = parser_generator.build(nn_archive)

    assert (
        len(parsers) == num_heads
    ), f"Expected {num_heads} parsers, got {len(parsers)}"


def test_host_only_flag(parser_generator: ParserGenerator):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(dai.NNModelDescription("luxonis/yolov6-nano", "RVC2"))
    )
    parsers = parser_generator.build(nn_archive, host_only=True)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], YOLOExtendedParser), "Expected YOLOExtendedParser"


def test_host_only_flag_unsupported(parser_generator: ParserGenerator):
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription("luxonis/mobilenet-ssd:300x300", "RVC2")
        )
    )
    with pytest.raises(ValueError):
        parser_generator.build(nn_archive, host_only=True)
