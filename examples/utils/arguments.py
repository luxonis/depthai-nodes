import argparse
from typing import Tuple


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser()
    parser.description = "General example script to run any model available in HubAI on DepthAI device. \
        All you need is a model slug of the model and the script will download the model from HubAI and create \
        the whole pipeline with visualizations. You also need a DepthAI device connected to your computer. \
        Currently, only RVC2 devices are supported."

    parser.add_argument(
        "-s",
        "--model_slug",
        help="slug of the model in HubAI.",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    return parser, args


def parse_model_slug(args: argparse.Namespace) -> Tuple[str, str]:
    """Parse the model slug from the arguments.

    Returns the model slug and model version slug.
    """
    model_slug = args.model_slug

    # parse the model slug
    if ":" not in model_slug:
        raise NameError(
            "Please provide the model slug in the format of 'model_slug:model_version_slug'"
        )

    model_slug_parts = model_slug.split(":")
    model_slug = model_slug_parts[0]
    model_version_slug = model_slug_parts[1]

    return model_slug, model_version_slug