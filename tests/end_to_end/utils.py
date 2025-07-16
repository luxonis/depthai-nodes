import os
import sys
from typing import Dict, List, Tuple

import depthai as dai
import requests

API_KEY = os.getenv("HUBAI_API_KEY", None)
HUBAI_TEAM_SLUG = os.getenv("HUBAI_TEAM_SLUG", None)

if not API_KEY:
    raise ValueError(
        "You must specify your HubAI API key (HUBAI_API_KEY) in order to get the model config."
    )

if not HUBAI_TEAM_SLUG:
    raise ValueError(
        "You must specify your HubAI team slug (HUBAI_TEAM_SLUG) in order to get the model config."
    )

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


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
        if inputs[0].layout == "NCHW":
            input_shape = inputs[0].shape[2:][::-1]
        elif inputs[0].layout == "NHWC":
            input_shape = inputs[0].shape[1:3][::-1]
        else:
            raise ValueError(
                "This model has an unsupported layout. Currently, only NCHW and NHWC layouts are supported."
            )
    except AttributeError:
        print(
            "This NN archive does not have an input shape. Please use NN archives that have input shapes."
        )
        exit(1)

    return input_shape


def get_num_inputs(nn_archive: dai.NNArchive) -> int:
    """Get the number of inputs of the model from the NN archive."""
    inputs = get_inputs_from_archive(nn_archive)

    return len(inputs)


def get_models() -> List[Dict]:
    """Get all the models from the ZOO that correspond to the HubAI team."""
    url = "https://easyml.cloud.luxonis.com/models/api/v1/models?is_public=true&limit=1000"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise ValueError(f"Failed to get models. Status code: {response.status_code}")
    response = response.json()

    valid_models = []

    for model in response:
        if model["is_public"] and model["team_slug"] == HUBAI_TEAM_SLUG:
            model_dict = {
                "name": model["name"],
                "slug": model["slug"],
                "model_id": model["id"],
            }
            valid_models.append(model_dict)

    return valid_models


def get_model_versions(models: List[Dict]) -> List[Dict]:
    """Get all the model versions from the ZOO that correspond to given models."""
    url = "https://easyml.cloud.luxonis.com/models/api/v1/modelVersions?is_public=True&limit=1000"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to get model versions. Status code: {response.status_code}")
        return None

    response = response.json()

    model_versions = []

    for version in response:
        for model in models:
            if version["model_id"] == model["model_id"]:
                model_version = {
                    "slug": model["slug"],
                    "version_slug": version["variant_slug"],
                    "model_id": model["model_id"],
                    "version_id": version["id"],
                    "name": model["name"],
                }
                model_versions.append(model_version)
                break

    return model_versions


def get_model_instances(model_versions: List[Dict]) -> List[Dict]:
    """Get all the model instances from the ZOO that correspond to the model
    versions."""
    url = "https://easyml.cloud.luxonis.com/models/api/v1/modelInstances?is_public=True&limit=1000"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to get model instances. Status code: {response.status_code}")
        return None

    response = response.json()

    model_instances = []

    already_checked = []

    for instance in response:
        for model in model_versions:
            if instance["model_version_id"] == model["version_id"]:
                if instance["model_version_id"] in already_checked:
                    continue
                model_instance = {
                    "slug": model["slug"],
                    "version_slug": model["version_slug"],
                    "model_id": model["model_id"],
                    "version_id": model["version_id"],
                    "instance_id": instance["id"],
                }
                model_instances.append(model_instance)
                already_checked.append(instance["model_version_id"])
                break

    return model_instances


def get_model_config_parser(model_instance_id: str) -> Dict:
    """Get the model config from the ZOO."""
    url = f"https://easyml.cloud.luxonis.com/models/api/v1/modelInstances/{model_instance_id}/config"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to get model config. Status code: {response.status_code}")
        return None

    return response.json()


def get_model_slugs_from_zoo() -> List[str]:
    """Get all the model slugs from the ZOO."""
    model_slugs = []
    models = get_models()
    model_versions = get_model_versions(models)

    for model in model_versions:
        model_slugs.append(f"{model['slug']}:{model['version_slug']}")

    return model_slugs


def find_slugs_from_zoo(parser: str) -> List[str]:
    """Find all model slugs that have the required parser."""
    models = get_models()
    model_versions = get_model_versions(models)
    model_instances = get_model_instances(model_versions)
    relevant_slugs = []
    already_checked = []

    for ix, instance in enumerate(model_instances):
        sys.stdout.write(
            f"\r[{ix + 1}/{len(model_instances)}] Checking model configs..."
        )
        sys.stdout.flush()
        model_config = get_model_config_parser(instance["instance_id"])
        if model_config is None:
            continue
        try:
            heads = model_config["model"]["heads"]
        except KeyError:
            print(
                f"Model {instance['slug']}:{instance['version_slug']} does not have heads."
            )
            continue

        try:
            parsers = [head["parser"] for head in heads]
        except KeyError:
            print(
                f"Model {instance['slug']}:{instance['version_slug']} does not have a parser."
            )
            continue

        if parser in parsers:
            if instance["version_id"] in already_checked:
                continue
            relevant_slugs.append(f"{instance['slug']}:{instance['version_slug']}")
            already_checked.append(instance["version_id"])

    print()
    return relevant_slugs
