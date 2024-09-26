from typing import List

import depthai as dai

from . import *
from .base_parser import BaseParser
from .utils import decode_head


class Parser(dai.node.ThreadedHostNode):
    """General interface for instantiating parsers based on the provided model
    archive."""

    def __init__(self):
        super().__init__()
        pass

    def build(
        self, nn_archive: dai.NNArchive, head__index: int = None
    ) -> List[BaseParser]:
        """Instantiates parsers based on the provided model archive.

        Attributes
        ----------
        nn_archive: dai.NNArchive
            NN Archive of the model.
        head__index: int
            Index of the head to be used for parsing. If not provided, each head will instantiate a separate parser.

        Returns
        -------
        parsers: List[BaseParser]
            List of instantiated parsers.
        """
        heads = nn_archive.getConfig().getConfigV1().model.heads

        if len(heads) == 0:
            raise ValueError("No heads defined in the NN Archive.")

        if head__index:
            heads = [heads[head__index]]
        parsers = []
        pipeline = self.getParentPipeline()

        for head in heads:
            parser_name = head.parser

            parser = globals().get(parser_name)

            if parser is None:
                raise ValueError(f"Parser {parser_name} not found in the parsers.")

            parser = parser()
            head = decode_head(head)
            parsers.append(pipeline.create(parser).build(head))

        return parsers
