import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

MANUAL_SCRIPT = Path(__file__).with_name("manual.py")


def get_platforms(platform: str) -> list[tuple[str, str]]:
    rvc2_ip = os.getenv("RVC2_IP", "")
    rvc4_ip = os.getenv("RVC4_IP", "")

    platforms = [(rvc2_ip, "RVC2"), (rvc4_ip, "RVC4")]
    if platform.lower() == "rvc2":
        return [(rvc2_ip, "RVC2")]
    if platform.lower() == "rvc4":
        return [(rvc4_ip, "RVC4")]
    return platforms


def run_pipeline(
    IP: str,
    ip_platform: str,
    nn_archive_path: str | None,
    model: str | None,
    *,
    native_parsers: bool = False,
) -> None:
    time.sleep(10)

    if not (nn_archive_path or model):
        raise ValueError("You have to pass either an NN archive path or a model")

    print(
        f"Testing model {model} from NN archive {nn_archive_path} on device with "
        f"IP {IP} ({ip_platform}); native parsers: {native_parsers}",
        flush=True,
    )

    command = [sys.executable, str(MANUAL_SCRIPT)]
    if model:
        command.extend(["-m", model])
    else:
        command.extend(["-nn", str(nn_archive_path)])
    if IP:
        command.extend(["-ip", IP])
    if native_parsers:
        command.append("--native-parsers")

    try:
        subprocess.run(command, check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 5:
            pytest.skip(f"Model {model} not supported on {ip_platform}.")
        if exc.returncode == 6:
            pytest.skip(f"Can't connect to the device with IP/mxid: {IP}")
        if exc.returncode == 7:
            pytest.skip(f"Couldn't find model {model} for {ip_platform} in the ZOO")
        if exc.returncode == 8:
            pytest.skip(f"The model {model} is not supported in this test.")
        if exc.returncode == 9:
            pytest.skip(f"Couldn't load model {model} from its NN archive.")
        raise RuntimeError("Pipeline crashed.") from exc
    except subprocess.TimeoutExpired:
        pytest.fail("Pipeline timeout.")
