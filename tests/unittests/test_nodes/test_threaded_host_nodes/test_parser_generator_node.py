import depthai as dai
import pytest

from depthai_nodes.node import ParserGenerator, YOLOExtendedParser
from tests.utils import PipelineMock


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
def parser_generator():
    pipeline = PipelineMock()
    return pipeline.create(ParserGenerator)


@pytest.mark.parametrize(
    "model_name",
    [
        "luxonis/yunet:320x240",
        "luxonis/vehicle-attributes-classification:72x72",
        "luxonis/yolov6-nano:r2-coco-512x288",
        "luxonis/mobilenet-ssd:300x300",
    ],
)
def test_parser_generator(parser_generator: ParserGenerator, model_name: str):
    nn_archive = get_model_archive(model_name)

    num_heads = len(nn_archive.getConfig().model.heads)

    parsers = parser_generator.build(nn_archive)

    assert (
        len(parsers) == num_heads
    ), f"Expected {num_heads} parsers, got {len(parsers)}"


def test_host_only_flag(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/yolov6-nano")
    parsers = parser_generator.build(nn_archive, hostOnly=True)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], YOLOExtendedParser), "Expected YOLOExtendedParser"


def test_detection_parser_generatior(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/yolov6-nano")
    parsers = parser_generator.build(nn_archive, hostOnly=False)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], dai.node.DetectionParser), "Expected DetectionParser"
    assert not parsers[0].runOnHost, "Expected runOnHost to be False"


def test_detection_segmentation_parser_generator(parser_generator: ParserGenerator):
    nn_archive = get_model_archive(
        "luxonis/yolov8-instance-segmentation-nano:coco-512x288"
    )
    parsers = parser_generator.build(nn_archive, hostOnly=False)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], dai.node.DetectionParser), "Expected DetectionParser"
    assert parsers[0].runOnHost, "Expected runOnHost to be True"


def test_host_only_flag_unsupported(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/mobilenet-ssd:300x300")
    with pytest.raises(ValueError):
        parser_generator.build(nn_archive, hostOnly=True)
