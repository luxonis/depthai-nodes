import os
from typing import List, Tuple

import depthai as dai
import requests
from slugs import SLUGS


def get_inputs_from_archive(nn_archive: dai.NNArchive) -> List:
    """Get all inputs from NN archive."""
    try:
        inputs = nn_archive.getConfig().model.inputs
    except AttributeError:
        print(
            "This NN archive does not have an input shape. Please use NN archives that have input shapes."
        )
        exit(1)

    return inputs


def get_input_shape(nn_archive: dai.NNArchive) -> Tuple[int, int]:
    """Get the input shape of the model from the NN archive."""
    inputs = get_inputs_from_archive(nn_archive)

    if len(inputs) > 1:
        raise ValueError(
            "This model has more than one input. Currently, only models with one input are supported."
        )

    try:
        input_shape = inputs[0].shape[2:][::-1]
    except AttributeError:
        print(
            "This NN archive does not have an input shape. Please use NN archives that have input shapes."
        )
        exit(1)

    return input_shape


def parse_model_slug(full_slug) -> Tuple[str, str]:
    """Parse the model slug into model_slug and model_version_slug."""
    if ":" not in full_slug:
        raise NameError(
            "Please provide the model slug in the format of 'model_slug:model_version_slug'"
        )
    model_slug_parts = full_slug.split(":")
    model_slug = model_slug_parts[0]
    model_version_slug = model_slug_parts[1]

    return model_slug, model_version_slug


def get_model_slugs_from_zoo():
    # For now we will use the slugs from the slugs.py file
    # because the ZOO API changed and DAI is not yet updated
    return SLUGS
    hubai_team_id = os.getenv("HUBAI_TEAM_ID", None)
    if not hubai_team_id:
        raise ValueError(
            "You must specify your HubAI team ID in order to get the models from ZOO."
        )

    url = "https://easyml.cloud.luxonis.com/models/api/v1/models?is_public=true&limit=1000"

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to get models. Status code: {response.status_code}")
    response = response.json()

    valid_models = []

    for model in response:
        if model["is_public"] and model["team_id"] == hubai_team_id:
            model_dict = {
                "name": model["name"],
                "slug": model["slug"],
                "model_id": model["id"],
            }
            valid_models.append(model_dict)

    # Get version slugs
    url = "https://easyml.cloud.luxonis.com/models/api/v1/modelVersions?is_public=True&limit=1000"

    response = requests.get(url).json()

    model_slugs = []

    for version in response:
        for model in valid_models:
            if version["model_id"] == model["model_id"]:
                model["version_slug"] = version["slug"]
                model_slugs.append(f"{model['slug']}:{model['version_slug']}")
                break

    return model_slugs
