from typing import Any


def decode_head(head) -> dict[str, Any]:
    """Decode head object into a dictionary containing configuration details.

    Args:
        head (``dai.nn_archive.v1.Head``): The head object to decode.

    Returns:
        A dictionary containing configuration details relevant to the head.
    """
    head_config = {}
    head_config["parser"] = head.parser
    head_config["outputs"] = head.outputs
    if head.metadata:
        head_config.update(head.metadata.extraParams)

    return head_config
