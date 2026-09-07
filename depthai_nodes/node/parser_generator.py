import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.node.parsers import *
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import decode_head


class ParserGenerator(dai.node.ThreadedHostNode):
    """General interface for instantiating parsers based on the provided model archive.

    The `build` method creates parsers based on the head information stored in the NN Archive. The method then returns a dictionary of these parsers.
    """

    _logger = get_logger(__name__)
    DETECTION_PARSERS = {"DetectionParser", "SSD", "YOLO", "YOLOExtendedParser"}
    HOST_PARSER_ALIASES = {"YOLO": "YOLOExtendedParser"}

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
        @param hostOnly: If True, use parser implementations from depthai-nodes.
            Otherwise, always use native DepthAI parser nodes.
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

            if hostOnly:
                parsers[index] = self._createHostParser(
                    pipeline,
                    parser_name,
                    head,
                    nnArchive.getConfig().model.inputs,
                )
                continue

            parser = pipeline.create(self._getNativeParserClass(parser_name))
            parser.setNNArchiveHead(head)
            self._setNativeParserInputSize(
                parser,
                head,
                nnArchive.getConfig().model.inputs,
            )

            is_detection_parser = parser_name in self.DETECTION_PARSERS
            run_native_parsers_on_host = is_rvc2_device and (
                not is_detection_parser
                or getattr(getattr(head, "metadata", None), "maskOutputs", None)
                is not None
            )
            if run_native_parsers_on_host:
                self._logger.warning(
                    f"Native {parser_name} detected with an RVC2 device. "
                    "Parsing will run on the host machine."
                )

            parsers[index] = parser

        return parsers

    def _createHostParser(self, pipeline, parser_name: str, head, model_inputs):
        parser_name = self.HOST_PARSER_ALIASES.get(parser_name, parser_name)
        parser_class = globals().get(parser_name)

        if parser_class is None or not isinstance(parser_class, type):
            raise ValueError(f"Parser {parser_name} is not available in depthai-nodes.")
        if not issubclass(parser_class, BaseParser):
            raise ValueError(
                f"Parser {parser_name} does not inherit from BaseParser class."
            )

        head_config = decode_head(head)
        head_config["model_inputs"] = [
            {"shape": model_input.shape, "layout": model_input.layout}
            for model_input in model_inputs
        ]
        return pipeline.create(parser_class).build(head_config)

    def _getNativeParserClass(self, parser_name: str):
        if parser_name in self.DETECTION_PARSERS:
            return dai.node.DetectionParser
        if parser_name == "SegmentationParser":
            return dai.node.SegmentationParser

        parser_class = getattr(dai.beta.node, parser_name, None)
        if parser_class is None or not isinstance(parser_class, type):
            raise ValueError(
                f"Parser {parser_name} is not available as a native DepthAI parser."
            )
        return parser_class

    @staticmethod
    def _setNativeParserInputSize(parser, head, model_inputs) -> None:
        if not hasattr(parser, "setInputSize") or not model_inputs:
            return

        head_config = decode_head(head)
        if head.parser in {"XFeatMonoParser", "XFeatStereoParser"}:
            if head_config.get("input_size") is not None:
                return

        model_input = model_inputs[0]
        shape = model_input.shape
        layout = str(model_input.layout).upper()

        if len(shape) != 4:
            raise ValueError(
                f"Cannot derive parser input size from model input shape {shape}."
            )
        if layout.endswith("NCHW"):
            width, height = shape[3], shape[2]
        elif layout.endswith("NHWC"):
            width, height = shape[2], shape[1]
        else:
            raise ValueError(
                f"Cannot derive parser input size from model input layout "
                f"{model_input.layout}."
            )

        parser.setInputSize(width, height)

    def run(self):
        """No-op required by ``dai.node.ThreadedHostNode``."""
        pass
