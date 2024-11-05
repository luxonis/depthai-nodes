import ast
import multiprocessing
import os
import time
from typing import List, Tuple

import depthai as dai
import pytest

from depthai_nodes.parsing_neural_network import ParsingNeuralNetwork


@pytest.fixture
def nn_archive_paths(request):
    return request.config.getoption("--nn_archive_path")


@pytest.fixture
def slugs(request):
    return request.config.getoption("--slug")


def parse_model_slug(full_slug) -> Tuple[str, str]:
    if ":" not in full_slug:
        raise NameError(
            "Please provide the model slug in the format of 'model_slug:model_version_slug'"
        )
    model_slug_parts = full_slug.split(":")
    model_slug = model_slug_parts[0]
    model_version_slug = model_slug_parts[1]

    return model_slug, model_version_slug


def get_parametrized_values(slugs: List[str], nn_archive_paths: List[str]):
    test_cases = []
    rvc2_ip = os.getenv("RVC2_IP", "")
    rvc4_ip = os.getenv("RVC4_IP", "")

    if slugs:
        slugs = ast.literal_eval(slugs)
        test_cases.extend(
            [
                (*IP, None, slug)
                for slug in slugs
                for IP in [(rvc2_ip, "RVC2"), (rvc4_ip, "RVC4")]
            ]
        )
    if nn_archive_paths:
        nn_archive_paths = ast.literal_eval(nn_archive_paths)
        test_cases.extend(
            [
                (*IP, nn_archive_path, None)
                for nn_archive_path in nn_archive_paths
                for IP in [(rvc2_ip, "RVC2"), (rvc4_ip, "RVC4")]
            ]
        )
    return test_cases


def pytest_generate_tests(metafunc):
    nn_archive_paths = metafunc.config.getoption("nn_archive_path")
    slugs = metafunc.config.getoption("slug")
    params = get_parametrized_values(slugs, nn_archive_paths)
    metafunc.parametrize("IP, ip_platform, nn_archive_path, slug", params)


def test_pipelines(IP: str, ip_platform: str, nn_archive_path, slug):
    time.sleep(2)
    subprocess = multiprocessing.Process(
        target=pipeline_test, args=[IP, nn_archive_path, slug]
    )
    subprocess.start()
    subprocess.join()
    if subprocess.exitcode != 0:
        if subprocess.exitcode == 5:
            pytest.skip(f"Model not supported on {ip_platform}.")
        else:
            raise RuntimeError("Pipeline crashed.")


def pipeline_test(IP: str, nn_archive_path: str = None, slug: str = None):
    if not (nn_archive_path or slug):
        raise ValueError("You have to pass either path to NNArchive or model slug")

    device = dai.Device(dai.DeviceInfo(IP))
    device_platform = device.getPlatform().name

    if slug:
        model_slug, model_version_slug = parse_model_slug(slug)
        nn_archive = dai.NNModelDescription(
            modelSlug=model_slug,
            modelVersionSlug=model_version_slug,
            platform=device_platform,
        )
        try:
            nn_archive_path = dai.getModelFromZoo(nn_archive)
        except Exception:
            device.close()
            exit(5)

    nn_archive = dai.NNArchive(nn_archive_path)

    model_platforms = [platform.name for platform in nn_archive.getSupportedPlatforms()]

    if device_platform not in model_platforms:
        device.close()
        exit(5)

    with dai.Pipeline(device) as pipeline:
        camera_node = pipeline.create(dai.node.Camera).build()

        nn_w_parser = pipeline.create(ParsingNeuralNetwork).build(
            camera_node, nn_archive
        )

        head_indices = nn_w_parser._parsers.keys()

        parser_output_queues = {
            i: nn_w_parser.getOutput(i).createOutputQueue() for i in head_indices
        }

        pipeline.start()

        while pipeline.isRunning():
            for head_id in parser_output_queues:
                parser_output = parser_output_queues[head_id].get()
                print(f"{head_id} - {type(parser_output)}")
            pipeline.stop()

        device.close()
