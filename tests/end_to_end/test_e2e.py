import ast
import os
import subprocess
import time
from typing import List

import pytest


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
    models: List[str], nn_archive_paths: List[str], platform: str
):
    test_cases = []
    rvc2_ip = os.getenv("RVC2_IP", "")
    rvc4_ip = os.getenv("RVC4_IP", "")

    platform = platform.lower()

    platforms = [(rvc2_ip, "RVC2"), (rvc4_ip, "RVC4")]
    if platform:
        if platform == "rvc2":
            platforms = [(rvc2_ip, "RVC2")]
        elif platform == "rvc4":
            platforms = [(rvc4_ip, "RVC4")]

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
    time.sleep(10)

    if not (nn_archive_path or model):
        raise ValueError("You have to pass either path to NNArchive or model model")

    try:
        if model:
            subprocess.run(
                f"python manual.py -m {model} {'-ip' if IP else ''} {IP}",
                shell=True,
                check=True,
                timeout=120,
            )
        else:
            subprocess.run(
                f"python manual.py -nn {nn_archive_path} {'-ip' if IP else ''} {IP}",
                shell=True,
                check=True,
                timeout=120,
            )
    except subprocess.CalledProcessError as e:
        if e.returncode == 5:
            pytest.skip(f"Model {model} not supported on {ip_platform}.")
        elif e.returncode == 6:
            pytest.skip(f"Can't connect to the device with IP/mxid: {IP}")
        elif e.returncode == 7:
            pytest.skip(f"Couldn't find model {model} for {ip_platform} in the ZOO")
        elif e.returncode == 8:
            pytest.skip(
                f"The model {model} is not supported in this test. (small input size, grayscale image, etc.)"
            )
        elif e.returncode == 9:
            pytest.skip(f"Couldn't load the model {model} from NN archive.")
        else:
            raise RuntimeError("Pipeline crashed.") from e
    except subprocess.TimeoutExpired:
        pytest.fail("Pipeline timeout.")
