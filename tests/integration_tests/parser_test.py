import ast
import subprocess
import time
from typing import List

import depthai as dai
import pytest


@pytest.fixture
def models(request):
    return request.config.getoption("--models")


def get_parametrized_values(models: List[str], parsers: List[str]) -> List[List[str]]:
    test_cases = []

    devices = dai.Device.getAllConnectedDevices()
    if not devices:
        pytest.skip("Couldn't find any devices.")

    devices = [
        ("RVC2" if "MYRIAD" in device.platform.name else "RVC4", device.getMxId())
        for device in devices
    ]

    if models:
        models = ast.literal_eval(models)
        parsers = ast.literal_eval(parsers)
        test_cases.extend(
            [
                (model, parsers[i], platform, mxid)
                for i, model in enumerate(models)
                for (platform, mxid) in devices
            ]
        )

    return test_cases


def pytest_generate_tests(metafunc):
    models = metafunc.config.getoption("models")
    parsers = metafunc.config.getoption("parsers")
    params = get_parametrized_values(models, parsers)
    metafunc.parametrize("models, parsers, platform, mxid", params)


def test_parser(models, parsers, platform, mxid):
    time.sleep(5)  # device needs some time to finish previous pipeline
    if not models:
        raise ValueError("You have to pass models")

    try:
        if models:
            subprocess.run(
                f"python manual.py -m {models} -ip {mxid}",
                shell=True,
                check=True,
                timeout=90,
            )
    except subprocess.CalledProcessError as e:
        if e.returncode == 6:
            pytest.skip("Couldn't connect to the default device.")
        elif e.returncode == 7:
            pytest.skip(f"Couldn't find model {models} in the ZOO")
        elif e.returncode == 8:
            pytest.skip(f"Couldn't load the model {models} from NN archive.")
        elif e.returncode == 9:
            pytest.skip(f"Couldn't extract the parser from the model {models}.")
        elif e.returncode == 10:
            pytest.skip(f"Couldn't load the tensors for the model {models}.")
        else:
            raise RuntimeError("Pipeline crashed.") from e
    except subprocess.TimeoutExpired:
        pytest.fail("Pipeline timeout.")
