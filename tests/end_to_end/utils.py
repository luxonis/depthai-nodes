import os

import requests


def get_model_slugs_from_zoo():
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
