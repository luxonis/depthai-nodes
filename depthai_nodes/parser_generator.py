from typing import Dict

import depthai as dai

from .ml.parsers import *
from .ml.parsers.base_parser import BaseParser
from .ml.parsers.utils import decode_head


class ParserGenerator(dai.node.ThreadedHostNode):
    """General interface for instantiating parsers based on the provided model archive.

    The `build` method creates parsers based on the head information stored in the NN Archive. The method then returns a dictionary of these parsers.
    """

    def __init__(self):
        super().__init__()

    def build(self, nn_archive: dai.NNArchive, head_index: int = None) -> Dict:
        """Instantiates parsers based on the provided model archive.

        Attributes
        ----------
        nn_archive: dai.NNArchive
            NN Archive of the model.
        head__index: int
            Index of the head to be used for parsing. If not provided, each head will instantiate a separate parser.

        Returns
        -------
        parsers: Dict[int : BaseParser]
            A dictionary of instantiated parsers.
        """
        heads = nn_archive.getConfig().model.heads
        indexes = range(len(heads))

        if len(heads) == 0:
            raise ValueError("No heads defined in the NN Archive.")

        if head_index:
            heads = [heads[head_index]]
            indexes = [head_index]

        parsers = {}
        pipeline = self.getParentPipeline()

        

        for index, head in zip(indexes, heads):
            parser_name = head.parser
            parser = globals().get(parser_name)

            if parser is None:
                raise ValueError(f"Parser {parser_name} not a valid parser class.")
            elif not issubclass(parser, BaseParser):
                raise ValueError(
                    f"Parser {parser_name} does not inherit from BaseParser class."
                )

            head = decode_head(head)
            parsers[index] = pipeline.create(parser).build(head)

        return parsers

    def run(self):
        pass
