import pytest

from tests.end_to_end.pipeline_runner import get_platforms, run_pipeline

pytestmark = pytest.mark.e2e

NATIVE_PARSER_SMOKE_MODELS = [
    "luxonis/yunet:320x240",
    "luxonis/yolov6-nano:r2-coco-512x288",
    "luxonis/mediapipe-selfie-segmentation:256x144",
]


def pytest_generate_tests(metafunc):
    platform = metafunc.config.getoption("platform")
    params = [
        (*device, model)
        for model in NATIVE_PARSER_SMOKE_MODELS
        for device in get_platforms(platform)
    ]
    metafunc.parametrize("IP, ip_platform, model", params)


def test_native_parsing_neural_network(IP: str, ip_platform: str, model: str):
    run_pipeline(
        IP,
        ip_platform,
        None,
        model,
        native_parsers=True,
    )
