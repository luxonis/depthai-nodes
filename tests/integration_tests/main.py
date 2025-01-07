import argparse
import os
import sys

import pytest
from config import parsers_slugs
from utils import download_test_files


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

    args = arg_parser.parse_args()
    models = args.model
    run_all = args.all
    parser = args.parser
    download = args.download

    if os.path.exists("nn_datas"):
        print()
        print("Folder `nn_datas` with test files already exists. Skipping download.")
        print()
    else:
        download_test_files()

    if download:
        download_test_files()

    print(f"Run all tests: {run_all}")

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
            print(f"Found models for parser {parser}: {models}")

    if not models:
        raise ValueError("No models provided")

    # Get keys from dictionary according to values in models
    parsers = [
        key
        for key, value in parsers_slugs.items()
        for model in models
        if model in value
    ]

    command = [
        "parser_test.py",
        f"--models={models}",
        f"--parsers={parsers}",
        "-v",
        "--tb=no",
        "-r a",
        "--log-cli-level=DEBUG",
        "--color=yes",
    ]
    exitcode = pytest.main(command)
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
