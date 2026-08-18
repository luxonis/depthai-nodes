import depthai as dai
import pytest

from depthai_nodes.node import (
    ClassificationParser,
    ParserGenerator,
    YOLOExtendedParser,
    YuNetParser,
)
from depthai_nodes.node import parsers as host_parsers
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
    ("model_name", "expected_parser", "warns_host_execution"),
    [
        ("luxonis/yunet:320x240", dai.beta.node.YuNetParser, True),
        (
            "luxonis/vehicle-attributes-classification:72x72",
            dai.beta.node.ClassificationParser,
            True,
        ),
        ("luxonis/yolov6-nano:r2-coco-512x288", dai.node.DetectionParser, False),
        ("luxonis/mobilenet-ssd:300x300", dai.node.DetectionParser, False),
    ],
)
def test_parser_generator(
    parser_generator: ParserGenerator,
    model_name: str,
    expected_parser,
    warns_host_execution: bool,
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(parser_generator._logger, "warning", warnings.append)
    nn_archive = get_model_archive(model_name)

    num_heads = len(nn_archive.getConfig().model.heads)

    parsers = parser_generator.build(nn_archive)

    assert (
        len(parsers) == num_heads
    ), f"Expected {num_heads} parsers, got {len(parsers)}"
    for parser in parsers.values():
        if expected_parser.__module__ == "depthai.beta.node":
            assert parser.node_type is expected_parser
        else:
            assert isinstance(parser, expected_parser)
        assert (
            not parser.runOnHost
        ), "Parser host execution is decided at pipeline start"
    assert bool(warnings) is warns_host_execution


@pytest.mark.parametrize(
    "parser_name",
    [
        parser_name
        for parser_name in host_parsers.__all__
        if parser_name != "BaseParser"
    ],
)
def test_all_depthai_nodes_parsers_have_native_mapping(
    parser_generator: ParserGenerator,
    parser_name: str,
):
    native_parser = parser_generator._getNativeParserClass(parser_name)

    if parser_name in {"DetectionParser", "YOLOExtendedParser"}:
        assert native_parser is dai.node.DetectionParser
    elif parser_name == "SegmentationParser":
        assert native_parser is dai.node.SegmentationParser
    else:
        assert native_parser is getattr(dai.beta.node, parser_name)


def test_host_only_flag(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/yolov6-nano")
    parsers = parser_generator.build(nn_archive, hostOnly=True)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], YOLOExtendedParser), "Expected YOLOExtendedParser"


@pytest.mark.parametrize(
    ("model_name", "expected_parser"),
    [
        ("luxonis/yunet:320x240", YuNetParser),
        (
            "luxonis/vehicle-attributes-classification:72x72",
            ClassificationParser,
        ),
    ],
)
def test_host_only_uses_depthai_nodes_parser(
    parser_generator: ParserGenerator,
    model_name: str,
    expected_parser,
):
    parsers = parser_generator.build(get_model_archive(model_name), hostOnly=True)

    assert all(isinstance(parser, expected_parser) for parser in parsers.values())


def test_detection_parser_generator(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/yolov6-nano")
    parsers = parser_generator.build(nn_archive, hostOnly=False)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], dai.node.DetectionParser), "Expected DetectionParser"
    assert not parsers[0].runOnHost, "Expected no host override before pipeline start"


def test_native_parser_runs_on_device_for_rvc4(parser_generator: ParserGenerator):
    pipeline = parser_generator.getParentPipeline()
    pipeline.getDefaultDevice()._platform = dai.Platform.RVC4

    parsers = parser_generator.build(
        get_model_archive("luxonis/yolov6-nano"),
        hostOnly=False,
    )

    assert not parsers[0].runOnHost, "Expected runOnHost to be False for RVC4"


def test_mask_output_detection_parser_warns_on_rvc2(
    parser_generator: ParserGenerator, monkeypatch
):
    warnings = []
    monkeypatch.setattr(parser_generator._logger, "warning", warnings.append)

    nn_archive = get_model_archive(
        "luxonis/yolov8-instance-segmentation-nano:coco-512x288"
    )
    parsers = parser_generator.build(nn_archive, hostOnly=False)
    assert parsers is not None, "Parsers should not be None"
    assert len(parsers) == 1, "Expected 1 parser"
    assert isinstance(parsers[0], dai.node.DetectionParser), "Expected DetectionParser"
    assert not parsers[0].runOnHost, "Expected no host override before pipeline start"
    assert warnings == [
        "Native YOLOExtendedParser detected with an RVC2 device. "
        "Parsing will run on the host machine."
    ]


def test_host_only_flag_unsupported(parser_generator: ParserGenerator):
    nn_archive = get_model_archive("luxonis/mobilenet-ssd:300x300")
    with pytest.raises(ValueError):
        parser_generator.build(nn_archive, hostOnly=True)
