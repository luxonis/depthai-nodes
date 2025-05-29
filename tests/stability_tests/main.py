import argparse
import os
import subprocess

import pytest
from config import parsers_slugs
from utils import download_test_files

from depthai_nodes.logging import get_logger

logger = get_logger(__name__)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="+",
        default="",
        help="Model from HubAI.",
    )
    arg_parser.add_argument("-all", action="store_true", help="Run all tests")
    arg_parser.add_argument(
        "-p", "--parser", default="", help="Name of the specific parser to test."
    )
    arg_parser.add_argument(
        "-d", "--download", action="store_true", help="Download test files"
    )
    arg_parser.add_argument(
        "--duration", type=int, default=10, help="Duration of the test in seconds"
    )

    args = arg_parser.parse_args()
    models = args.model
    run_all = args.all
    parser = args.parser
    download = args.download
    duration = args.duration

    if os.path.exists("nn_datas"):
        logger.info(
            "Folder `nn_datas` with test files already exists. Skipping download."
        )
    else:
        download_test_files()

    if download:
        download_test_files()

    logger.info(f"Run all tests: {run_all}")
    logger.info(f"Duration of each test: {duration} seconds")

    if run_all and models:
        raise ValueError("You can't pass both -all and --model")

    if run_all:
        models = list(parsers_slugs.values())
        models = [model for sublist in models for model in sublist]

    if parser:
        models = parsers_slugs.get(parser, None)
        if not models:
            raise ValueError(f"No models found for parser {parser}")
        else:
            logger.info(f"Found models for parser {parser}: {models}")

    if not models:
        raise ValueError("No models provided")

    # Get keys from dictionary according to values in models
    parsers = [
        key
        for key, value in parsers_slugs.items()
        for model in models
        if model in value
    ]

    models = repr(models)
    parsers = repr(parsers)
    try:
        subprocess.run(
            f'pytest run_parser_test.py --models "{models}" --parsers "{parsers}" --duration {duration} -v --tb=short -r a --log-cli-level=DEBUG --color=yes -s',
            shell=True,
            check=True,
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

    if run_all:
        try:
            subprocess.run(
                f'pytest ../ --duration {duration} -v --tb=short -r a --log-cli-level=DEBUG --color=yes -s -k "not test_creators and not test_messages"',
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Pipeline crashed.") from e

    logger.info("All tests passed.")


if __name__ == "__main__":
    main()
