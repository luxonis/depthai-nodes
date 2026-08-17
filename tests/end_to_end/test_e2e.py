import ast

import pytest

from tests.end_to_end.pipeline_runner import get_platforms, run_pipeline


@pytest.fixture
def nn_archive_paths(request):
    return request.config.getoption("--nn_archive_path")


@pytest.fixture
def models(request):
    return request.config.getoption("--model")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")


def get_parametrized_values(
    models: list[str], nn_archive_paths: list[str], platform: str
):
    test_cases = []
    platforms = get_platforms(platform)

    if models:
        models = ast.literal_eval(models)
        test_cases.extend([(*IP, None, model) for model in models for IP in platforms])
    if nn_archive_paths:
        nn_archive_paths = ast.literal_eval(nn_archive_paths)
        test_cases.extend(
            [
                (*IP, nn_archive_path, None)
                for nn_archive_path in nn_archive_paths
                for IP in platforms
            ]
        )
    return test_cases


def pytest_generate_tests(metafunc):
    nn_archive_paths = metafunc.config.getoption("nn_archive_path")
    models = metafunc.config.getoption("model")
    platform = metafunc.config.getoption("platform")
    params = get_parametrized_values(models, nn_archive_paths, platform)
    metafunc.parametrize("IP, ip_platform, nn_archive_path, model", params)


def test_pipelines(IP: str, ip_platform: str, nn_archive_path, model):
    run_pipeline(IP, ip_platform, nn_archive_path, model)
