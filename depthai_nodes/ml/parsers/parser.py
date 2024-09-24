from typing import Dict, List, Union

import depthai as dai


class Parser(dai.node.ThreadedHostNode):
    """Base class for all parsers."""

    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.out = self.createOutput()

        self.head_config: Dict = {}

    def build(self, heads: Union[List, Dict], head_name: str = ""):
        """Initial build method for all parsers. The method sets the head configuration
        for the parser.

        Attributes
        ----------
        heads : Union[List, Dict]
            List of all archive head objects: [<depthai.nn_archive.v1.Head object>, ...]
            or a dictionary containing the head metadata.
        head_name : str
            The name of the head to use. If multiple heads are available in the archive, the name must be specified.

        Returns
        -------
        Parser
            Returns the parser object with the head configuration set.
        """

        if isinstance(heads, list):
            if len(heads) == 0:
                raise ValueError("No heads available in the nn_archive.")

            head = heads[0]

            if len(heads) > 1 and head_name == "":
                current_parser = self.__class__.__name__
                parser_names_in_archive = [head.parser for head in heads]
                num_matches = parser_names_in_archive.count(current_parser)
                if num_matches == 0:
                    raise ValueError(
                        f"No heads available for {current_parser} in the nn_archive."
                    )
                elif num_matches == 1:
                    head = [head for head in heads if head.parser == current_parser][0]
                else:
                    raise ValueError(
                        f"Multiple heads with parser= {current_parser} detected, please specify a head name."
                    )

            if head_name != "":
                head_candidates = [
                    head
                    for head in heads
                    if head.metadata.extraParams["name"] == head_name
                ]
                if len(head_candidates) == 0:
                    raise ValueError(
                        f"No head with name {head_name} specified in nn_archive."
                    )
                if len(head_candidates) > 1:
                    raise ValueError(
                        f"Multiple heads with name {head_name} found in nn_archive, please specify a unique name."
                    )
                head = head_candidates[0]

            parser_name = head.parser
            metadata = head.metadata
            outputs = head.outputs

            if outputs is None:
                raise ValueError(
                    f"{head_name} head does not have any outputs specified."
                )

            head_dictionary = {}
            head_dictionary["parser"] = parser_name
            head_dictionary["outputs"] = outputs
            if metadata is not None:
                head_dictionary.update(metadata.extraParams)
            self.head_config = head_dictionary

        else:
            self.head_config = heads

        return self
