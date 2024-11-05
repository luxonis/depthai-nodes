import argparse
import os
from typing import Tuple

import depthai as dai
import pytest
from utils import get_model_slugs_from_zoo


def parse_model_slug(full_slug) -> Tuple[str, str]:
    if ":" not in full_slug:
        raise NameError(
            "Please provide the model slug in the format of 'model_slug:model_version_slug'"
        )
    model_slug_parts = full_slug.split(":")
    model_slug = model_slug_parts[0]
    model_version_slug = model_slug_parts[1]

    return model_slug, model_version_slug


def find_slugs(parser: str):
    relevant_slugs = []
    print("Downloading NN archives from the ZOO")
    slugs = get_model_slugs_from_zoo()
    n = len(slugs)
    for i, slug in enumerate(slugs):
        if "dm-count" in slug:
            print(f"[{i+1}/{n}] Skipping {slug} ...")
            continue
        print(f"[{i+1}/{n}] Downloading {slug} ...")
        model_slug, version_slug = parse_model_slug(slug)
        model_desc = dai.NNModelDescription(
            modelSlug=model_slug, modelVersionSlug=version_slug, platform="RVC2"
        )
        nn_archive_path = None
        try:
            nn_archive_path = dai.getModelFromZoo(model_desc)
        except Exception:
            model_desc = dai.NNModelDescription(
                modelSlug=model_slug, modelVersionSlug=version_slug, platform="RVC4"
            )
            try:
                nn_archive_path = dai.getModelFromZoo(model_desc, useCached=True)
            except Exception as e:
                raise ValueError(f"Couldn't find model {slug} in the ZOO") from e

        try:
            nn_archive = dai.NNArchive(nn_archive_path)
            for head in nn_archive.getConfig().model.heads:
                if head.parser == parser:
                    relevant_slugs.append(slug)
                    break
        except Exception as e:
            print(e)

        print(f"[{i+1}/{n}] Successfully downloaded {slug}!")
    return relevant_slugs


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

    args = arg_parser.parse_args()
    nn_archive_path = args.nn_archive_path  # it is a list of paths
    slug = args.slug
    run_all = args.all
    parser = args.parser
    print(f"Run all tests: {run_all}")
    print(f"RVC2 IP: {os.getenv('RVC2_IP', '')}")
    print(f"RVC4 IP: {os.getenv('RVC4_IP', '')}")

    if run_all and (nn_archive_path or slug):
        raise ValueError("You can't pass both -all and -nn_archive_path or -slug")

    if run_all:
        slug = get_model_slugs_from_zoo()

    if parser:
        slug = find_slugs(parser)
        if len(slug) == 0:
            raise ValueError(f"No models found for parser {parser}")
        else:
            print(f"Found model slugs for parser {parser}: {slug}")

    if not nn_archive_path and not slug:
        raise ValueError("You have to pass either path to NNArchive or model slug")

    slug = [f"{s}" for s in slug]

    if slug:
        pytest.main(
            [
                "test_e2e.py",
                f"--slug={slug}",
                "-v",
            ]
        )
        return

    pytest.main(
        [
            "test_e2e.py",
            f"--nn_archive_path={nn_archive_path}",
            "-v",
        ]
    )


if __name__ == "__main__":
    main()
