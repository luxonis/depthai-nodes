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
        nnArchive: dai.NNArchive,
        headIndex: Optional[int] = None,
        hostOnly: bool = False,
    ) -> Dict:
        """Instantiate parser nodes for the supplied model archive.

        Parameters
        ----------
        nnArchive
            Model archive describing the parser configuration.
        headIndex
            Optional model head index to instantiate. If omitted, parsers are
            created for all heads.
        hostOnly
            If ``True``, prefer host-side parser implementations where available.

        Returns
        -------
        Dict
            Mapping of model head index to parser node.
        """

        heads: List = nnArchive.getConfig().model.heads  # type: ignore

        indexes = range(len(heads))

        if len(heads) == 0:
            raise ValueError("No heads defined in the NN Archive.")

        if headIndex is not None:
            heads = [heads[headIndex]]
            indexes = [headIndex]

        parsers = {}
        pipeline = self.getParentPipeline()

        for index, head in zip(indexes, heads):
            parser_name = head.parser

            if parser_name in self.DEVICE_PARSERS:
                if hostOnly:
                    parser_name = self._getHostParserName(parser_name)
                else:
                    parser = pipeline.create(dai.node.DetectionParser)
                    parser.setNNArchive(nnArchive)
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
            for input in nnArchive.getConfig().model.inputs:
                head_config["model_inputs"].append(
                    {"shape": input.shape, "layout": input.layout}
                )
            parsers[index] = pipeline.create(parser).build(head_config)

        return parsers

    def run(self):
        """No-op required by ``dai.node.ThreadedHostNode``."""
        pass

    def _getHostParserName(self, parser_name: str) -> str:
        if parser_name == "YOLO":
            return YOLOExtendedParser.__name__  # noqa: F405
        else:
            raise ValueError(
                f"Parser {parser_name} is not supported for host only mode."
            )
