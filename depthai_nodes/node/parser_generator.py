from typing import Dict, List, Optional

import depthai as dai

from depthai_nodes.node.parsers import *
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import decode_head


class ParserGenerator(dai.node.ThreadedHostNode):
    """General interface for instantiating parsers based on the provided model archive.

    The `build` method creates parsers based on the head information stored in the NN Archive. The method then returns a dictionary of these parsers.
    """

    DEVICE_PARSERS = ["YOLO", "SSD"]

    def build(
        self,
        nn_archive: dai.NNArchive,
        head_index: Optional[int] = None,
        host_only: bool = False,
    ) -> Dict:
        """Instantiates parsers based on the provided model archive.

        @param nn_archive: NN Archive of the model.
        @type nn_archive: dai.NNArchive
        @param head_index: Index of the head to be used for parsing. If not provided,
            each head will instantiate a separate parser.
        @type head_index: Optional[int]
        @param host_only: If True, only host parsers will be instantiated.
        @type host_only: bool
        @return: A dictionary of instantiated parsers.
        @rtype: Dict[int, BaseParser]
        """

        heads: List = nn_archive.getConfig().model.heads  # type: ignore

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

            if parser_name in self.DEVICE_PARSERS:
                if host_only:
                    parser_name = self._getHostParserName(parser_name)
                else:
                    parser = pipeline.create(dai.node.DetectionParser)
                    parser.setNNArchive(nn_archive)
                    parsers[index] = parser
                    continue

            parser = globals().get(parser_name)

            if parser is None:
                raise ValueError(f"Parser {parser_name} not a valid parser class.")
            elif not issubclass(parser, BaseParser):
                raise ValueError(
                    f"Parser {parser_name} does not inherit from BaseParser class."
                )

            head_config = decode_head(head)
            head_config["model_inputs"] = []
            for input in nn_archive.getConfig().model.inputs:
                head_config["model_inputs"].append(
                    {"shape": input.shape, "layout": input.layout}
                )
            parsers[index] = pipeline.create(parser).build(head_config)

        return parsers

    def _getHostParserName(self, parser_name: str) -> str:
        if parser_name == "YOLO":
            return YOLOExtendedParser.__name__  # noqa: F405
        else:
            raise ValueError(
                f"Parser {parser_name} is not supported for host only mode."
            )

    def run(self):
        pass
