from typing import Dict, List, Union

import depthai as dai


class Parser(dai.node.ThreadedHostNode):
    """Base class for all parsers."""

    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.out = self.createOutput()

        self.head_configs: Dict = {}

    def build(self, head_metadata: Union[List, Dict], name: str = ""):
        """Initial build method for all parsers."""

        if isinstance(head_metadata, list):
            if len(head_metadata) == 0:
                raise ValueError("No heads parsed from archive")
            if len(head_metadata) > 1 and name == "":
                raise ValueError("Multiple heads detected, please specify head name")
            head = head_metadata[0]
            if name != "":
                head_candidates = [
                    head
                    for head in head_metadata
                    if head.metadata.extraParams["name"] == name
                ]
                if len(head_candidates) == 0:
                    raise ValueError("Head name not found in archive")
                if len(head_candidates) > 1:
                    raise ValueError(
                        "Multiple heads with the same name found in archive, please specify a unique head name"
                    )
                head = head_candidates[0]

            parser_name = head.parser
            metadata = head.metadata
            outputs = head.outputs

            if parser_name is None:
                raise ValueError("Head does not have a parser specified.")
            if outputs is None:
                raise ValueError("Head does not have any outputs.")
            if metadata is None:
                raise ValueError("Head does not have any metadata.")

            head_dictionary = {}
            head_dictionary["parser"] = parser_name
            head_dictionary["outputs"] = outputs
            head_dictionary.update(metadata.extraParams)
            self.head_configs = head_dictionary

        else:
            self.head_configs = head_metadata

        return self
