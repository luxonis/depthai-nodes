import argparse
from typing import Tuple


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser()
    parser.description = "General example script to run any model available in HubAI on DepthAI device. \
        All you need is a model slug of the model and the script will download the model from HubAI and create \
        the whole pipeline with visualizations. You also need a DepthAI device connected to your computer. \
        Currently, only RVC2 devices are supported. If using OAK-D Lite, please set the FPS limit to 28."

    parser.add_argument(
        "-s",
        "--slug",
        help="slug of the model in HubAI.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=30.0,  # default DepthAI FPS value
        type=float,
    )

    args = parser.parse_args()

    return parser, args
