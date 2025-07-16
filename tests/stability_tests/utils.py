import os
from typing import List, Tuple

import depthai as dai
from b2sdk.api import B2Api


def extract_parser(nn_archive: dai.NNArchive) -> str:
    """Extract the parser from the first head in NN archive."""
    try:
        parser = nn_archive.getConfig().model.heads[0].parser
    except AttributeError:
        print(
            "This NN archive does not have a parser. Please use NN archives that have parsers."
        )
        exit(1)

    return parser


def extract_main_slug(model_slug: str) -> str:
    """Extract the main slug from the model slug."""
    return model_slug.split("/")[1].split(":")[0]


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


def download_test_files():
    """Download test files from Backblaze B2.

    The B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set in the environment
    variables. The files are downloaded to the nn_datas folder.
    """
    b2_application_key_id = os.getenv("B2_APPLICATION_KEY_ID", None)
    b2_application_key = os.getenv("B2_APPLICATION_KEY", None)

    if not b2_application_key_id or not b2_application_key:
        raise ValueError(
            "B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set in the environment variables."
        )

    api = B2Api()
    api.authorize_account(
        application_key_id=b2_application_key_id, application_key=b2_application_key
    )

    bucket = api.get_bucket_by_name("luxonis")

    files = []

    if not os.path.exists("nn_datas"):
        os.mkdir("nn_datas")

    for file_version, folder_path in bucket.ls("depthai-nodes/nn_datas"):
        if not folder_path:
            continue
        if folder_path.endswith("/"):  # check if the file is a folder
            folder_name = folder_path.split("/")[-2]
            print(f"Downloading test files for {folder_name}.")
            for file_version, _ in bucket.ls(folder_path):
                if file_version.file_name.endswith(".bzEmpty"):
                    continue
                files.append(
                    {"filename": file_version.file_name, "id": file_version.id_}
                )
                f = bucket.download_file_by_id(file_version.id_)
                filename = file_version.file_name
                if not os.path.exists("nn_datas/" + folder_name):
                    os.mkdir("nn_datas/" + folder_name)
                filename = os.path.join(
                    "nn_datas", folder_name, filename.split("/")[-1]
                )
                f.save_to(filename)
