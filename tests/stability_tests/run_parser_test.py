import ast
import pickle
from typing import List

import depthai as dai
import pytest

from tests.utils import PipelineMock

from .check_messages import check_output
from .utils import extract_main_slug, extract_parser


@pytest.fixture(scope="session")
def models(request):
    return request.config.getoption("--models")


@pytest.fixture(scope="session")
def duration(request):
    return request.config.getoption("--duration")


@pytest.fixture(scope="session")
def parser_generator():
    from depthai_nodes.node.parser_generator import ParserGenerator

    pipeline = PipelineMock()
    return pipeline.create(ParserGenerator)


def get_parametrized_values(
    models: List[str], parsers: List[str], duration: int
) -> List[List[str]]:
    test_cases = []

    if duration:
        duration = int(duration)

    if models:
        models = ast.literal_eval(models)
        parsers = ast.literal_eval(parsers)
        test_cases.extend(
            [(model, parsers[i], duration) for i, model in enumerate(models)]
        )

    return test_cases


def pytest_generate_tests(metafunc):
    models = metafunc.config.getoption("models")
    parsers = metafunc.config.getoption("parsers")
    duration = metafunc.config.getoption("duration")
    params = get_parametrized_values(models, parsers, duration)
    metafunc.parametrize("model, parser_name, duration", params)


def load_tensors(model: str, parser: str) -> dai.NNData:
    model = extract_main_slug(model)
    nn_data = dai.NNData()
    with open(f"nn_datas/{parser}/{model}.pkl", "rb") as f:
        data = pickle.load(f)
        for key, value in data.items():
            nn_data.addTensor(str(key), value.tolist())
        return nn_data


def test_parser(parser_generator, model: str, parser_name: str, duration: int):
    # Get the model from the HubAI
    try:
        model_description = dai.NNModelDescription(model=model, platform="RVC2")
        archive_path = dai.getModelFromZoo(model_description)
    except Exception:
        try:
            model_description = dai.NNModelDescription(model=model, platform="RVC4")
            archive_path = dai.getModelFromZoo(model_description)
        except Exception as e:
            print(f"Error: {e}")
            exit(7)

    try:
        nn_archive = dai.NNArchive(archive_path)
    except Exception as e:
        print(f"Error: {e}")
        exit(8)

    try:
        parser_name = extract_parser(nn_archive)
    except Exception as e:
        print(e)
        exit(9)

    # Create and build parser
    parser = parser_generator.build(nn_archive=nn_archive)[0]
    parser.input._queue.duration = duration

    # Load and send test data
    try:
        nn_data = load_tensors(model, parser_name)
    except Exception as e:
        print(e)
        exit(10)

    parser.input.send(nn_data)
    parser.out.createOutputQueue(check_output, model, parser_name)  # must be created

    parser.run()
