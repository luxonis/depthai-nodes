import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.node.parsers import *
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import decode_head
from depthai_nodes.node.parsers.utils.yolo import YOLOSubtype, resolve_yolo_strides


class ParserGenerator(dai.node.ThreadedHostNode):
    """General interface for instantiating parsers based on the provided model archive.

    The `build` method creates parsers based on the head information stored in the NN Archive. The method then returns a dictionary of these parsers.
    """

    _logger = get_logger(__name__)
    DEVICE_PARSERS = ["YOLO", "SSD"]
    DAI_SUPPORTED_YOLO_SUBTYPES = [
        YOLOSubtype.V3,
        YOLOSubtype.V3T,
        YOLOSubtype.V3UT,
        YOLOSubtype.V5,
        YOLOSubtype.V5U,
        YOLOSubtype.V6,
        YOLOSubtype.V6R1,
        YOLOSubtype.V6R2,
        YOLOSubtype.V7,
        YOLOSubtype.V8,
        YOLOSubtype.V9,
        YOLOSubtype.V10,
        YOLOSubtype.V11,
        YOLOSubtype.V26,
        YOLOSubtype.P,
        YOLOSubtype.GOLD,
    ]

    def build(
        self,
        nnArchive: dai.NNArchive,
        headIndex: int | None = None,
        hostOnly: bool = False,
    ) -> dict:
        """Instantiate parser nodes for the supplied model archive.

        @param nnArchive: Model archive describing the parser configuration.
        @type nnArchive: dai.NNArchive
        @param headIndex: Optional model head index to instantiate. If omitted, parsers
            are created for all heads.
        @type headIndex: int | None
        @param hostOnly: If True, prefer host-side parser implementations where
            available.
        @type hostOnly: bool
        @return: Mapping of model head index to parser node.
        @rtype: dict
        """

        heads: list = nnArchive.getConfig().model.heads  # type: ignore

        indexes = range(len(heads))

        if len(heads) == 0:
            raise ValueError("No heads defined in the NN Archive.")

        if headIndex is not None:
            heads = [heads[headIndex]]
            indexes = [headIndex]

        parsers = {}
        pipeline = self.getParentPipeline()
        is_rvc2_device = pipeline.getDefaultDevice().getPlatform() == dai.Platform.RVC2

        for index, head in zip(indexes, heads):
            parser_name = head.parser

            if parser_name in self.DEVICE_PARSERS:
                if hostOnly:
                    parser_name = self._getHostParserName(parser_name)
                elif parser_name == "YOLO" and self._has_non_default_yolo_strides(head):
                    parser_name = self._getHostParserName(parser_name)
                else:
                    parser = pipeline.create(dai.node.DetectionParser)
                    parser.setNNArchive(nnArchive)
                    parsers[index] = parser
                    continue
            elif parser_name == "YOLOExtendedParser":
                yolo_subtype_str = head.metadata.subtype
                if yolo_subtype_str is not None:
                    yolo_subtype = YOLOSubtype(yolo_subtype_str.lower())
                    if (
                        yolo_subtype in self.DAI_SUPPORTED_YOLO_SUBTYPES
                        and not self._has_non_default_yolo_strides(head, yolo_subtype)
                    ):
                        parser = pipeline.create(dai.node.DetectionParser)
                        parser.setNNArchive(nnArchive)
                        if head.metadata.maskOutputs is not None and is_rvc2_device:
                            self._logger.warning(
                                "Segmentation based model detected with RVC2 device. Parsing will be done on the host machine."
                            )
                            parser.setRunOnHost(True)
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

    @staticmethod
    def _has_non_default_yolo_strides(
        head,
        subtype: YOLOSubtype | None = None,
    ) -> bool:
        head_config = decode_head(head)
        strides = head_config.get("strides")
        if strides is None:
            return False

        metadata = getattr(head, "metadata", None)

        if subtype is None:
            yolo_subtype_str = getattr(metadata, "subtype", None)
            if yolo_subtype_str is None:
                subtype = YOLOSubtype.DEFAULT
            else:
                try:
                    subtype = YOLOSubtype(yolo_subtype_str.lower())
                except ValueError:
                    return True

        yolo_outputs = getattr(metadata, "yoloOutputs", None)
        outputs = (
            yolo_outputs if yolo_outputs is not None else head_config.get("outputs")
        )
        num_outputs = len(outputs or [])

        try:
            resolved_strides = resolve_yolo_strides(
                strides,
                subtype,
                num_outputs,
            )
        except ValueError:
            return True

        default_strides = resolve_yolo_strides(
            None,
            subtype,
            num_outputs,
        )
        return resolved_strides != default_strides

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
