import argparse
import os
import sys

import pytest
from utils import find_slugs_from_zoo, get_model_slugs_from_zoo


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-nn",
        "--nn_archive_path",
        nargs="+",
        type=str,
        default="",
        help="Path(s) to the NNArchive.",
    )
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
        "--platform",
        type=str,
        default="",
        help="RVC platform to run the tests on. Default is both.",
    )

    args = arg_parser.parse_args()
    nn_archive_path = args.nn_archive_path  # it is a list of paths
    model = args.model
    run_all = args.all
    parser = args.parser
    rvc_platform = "both" if args.platform == "" else args.platform
    print(f"Run all tests: {run_all}")
    print(f"RVC2 IP: {os.getenv('RVC2_IP', '')}")
    print(f"RVC4 IP: {os.getenv('RVC4_IP', '')}")
    print(f"RVC platform: {'RVC2 & RVC4' if rvc_platform == '' else rvc_platform}")

    if run_all and (nn_archive_path or model):
        raise ValueError("You can't pass both -all and -nn_archive_path or -model")

    if run_all:
        model = get_model_slugs_from_zoo()

    if parser:
        model = find_slugs_from_zoo(parser)
        if len(model) == 0:
            raise ValueError(f"No models found for parser {parser}")
        else:
            print(f"Found models for parser {parser}: {model}")

    if not nn_archive_path and not model:
        raise ValueError("You have to pass either path to NNArchive or model")

    model = [f"{m}" for m in model]

    command = [
        "test_e2e.py",
        f"--nn_archive_path={nn_archive_path}",
        f"--platform={rvc_platform}",
        "-v",
        "--tb=short",
        "-r a",
        "--log-cli-level=DEBUG",
        "--color=yes",
    ]

    if model:
        command = [
            "test_e2e.py",
            f"--model={model}",
            f"--platform={rvc_platform}",
            "-v",
            "--tb=short",
            "-r a",
            "--log-cli-level=DEBUG",
            "--color=yes",
        ]

    exit_code = pytest.main(command)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
