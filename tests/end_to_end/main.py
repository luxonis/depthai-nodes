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
        "-s",
        "--slug",
        type=str,
        nargs="+",
        default="",
        help="Slug(s) of the model from HubAI.",
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
    slug = args.slug
    run_all = args.all
    parser = args.parser
    rvc_platform = "both" if args.platform == "" else args.platform
    print(f"Run all tests: {run_all}")
    print(f"RVC2 IP: {os.getenv('RVC2_IP', '')}")
    print(f"RVC4 IP: {os.getenv('RVC4_IP', '')}")
    print(f"RVC platform: {'RVC2 & RVC4' if rvc_platform == '' else rvc_platform}")

    if run_all and (nn_archive_path or slug):
        raise ValueError("You can't pass both -all and -nn_archive_path or -slug")

    if run_all:
        slug = get_model_slugs_from_zoo()

    if parser:
        slug = find_slugs_from_zoo(parser)
        if len(slug) == 0:
            raise ValueError(f"No models found for parser {parser}")
        else:
            print(f"Found model slugs for parser {parser}: {slug}")

    if not nn_archive_path and not slug:
        raise ValueError("You have to pass either path to NNArchive or model slug")

    slug = [f"{s}" for s in slug]

    command = [
        "test_e2e.py",
        f"--nn_archive_path={nn_archive_path}",
        f"--platform={rvc_platform}",
        "-v",
        "--tb=no",
        "-r s",
        "--log-cli-level=DEBUG",
        "--color=yes",
    ]

    if slug:
        command = [
            "test_e2e.py",
            f"--slug={slug}",
            f"--platform={rvc_platform}",
            "-v",
            "--tb=no",
            "-r s",
            "--log-cli-level=DEBUG",
            "--color=yes",
        ]

    exit_code = pytest.main(command)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
