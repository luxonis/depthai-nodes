from typing import Any


def decode_head(head) -> dict[str, Any]:
    """Decode head object into a dictionary containing configuration details.

    @param head: The head object to decode.
    @type head: dai.nn_archive.v1.Head
    @return: A dictionary containing configuration details relevant to the head.
    @rtype: dict[str, Any]
    """
    head_config = {}
    head_config["parser"] = head.parser
    head_config["outputs"] = head.outputs
    if head.metadata:
        head_config.update(head.metadata.extraParams)

    return head_config
