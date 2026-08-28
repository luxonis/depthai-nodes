from pathlib import Path
from typing import TypeVar, overload

import depthai as dai

from depthai_nodes.logging import get_logger
from depthai_nodes.node.parser_generator import ParserGenerator
from depthai_nodes.node.parsers import BaseParser

TParser = TypeVar("TParser", bound=BaseParser | dai.DeviceNode)


class ParsingNeuralNetwork(dai.node.ThreadedHostNode):
    Propeties = dai.node.NeuralNetwork.Properties
    """Node that wraps the NeuralNetwork node and adds parsing capabilities. A
    NeuralNetwork node is created with it's appropriate parser nodes for each model
    head. Parser nodes are chosen based on the supplied NNArchive.

    Attributes
    ----------
    input : Node.Input
        Neural network input.
    inputs : Node.InputMap
        Neural network inputs.
    out : Node.Output
        Neural network output. Can be used only when there is exactly one model head. Otherwise, getOutput method must be used.
    outputs: Node.Output
        Neural network output having dai.MessageGroup as a payload which contains outputs of all model heads and can be accessed as a dictionary with str(model head index) as a key. Can be used only when there is at least two model heads. Otherwise, out property must be used.
    passthrough : Node.Output
        Neural network passthrough.
    passthroughs : Node.OutputMap
        Neural network passthroughs.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the wrapper and create the internal neural-network node."""
        super().__init__(*args, **kwargs)
        self._nn = self.getParentPipeline().create(dai.node.NeuralNetwork)
        self._parsers: dict[int, BaseParser] = {}
        self._internal_sync: dai.node.Sync | None = None
        self._input_adapter: dai.DeviceNode | None = None
        self._logger = get_logger(__name__)
        self._logger.debug("ParsingNeuralNetwork initialized")

    @property
    def input(self) -> dai.Node.Input:
        """Return the primary neural-network input."""
        return self._nn.input

    @property
    def inputs(self) -> dai.Node.InputMap:
        """Return the neural-network input map."""
        return self._nn.inputs

    @property
    def out(self) -> dai.Node.Output:
        """Return the single parser output when the model has exactly one head."""
        self._checkNNArchive()
        if len(self._parsers) != 1:
            raise RuntimeError(
                f"Property out is only available when there is exactly one model head. \
                               The model has {self._getModelHeadsLen()} heads. Use {self.getOutput.__name__} method or outputs property instead."
            )
        return list(self._parsers.values())[0].out

    @property
    def outputs(self) -> dai.Node.Output:
        """Neural network output having dai.MessageGroup as a payload which contains
        outputs of all model heads and can be accessed as a dictionary with str(model
        head index) as a key.

        Can be used only when there are at least two model heads. Otherwise, out
        property must be used.
        """
        if self._internal_sync is None:
            raise RuntimeError(
                f"ParsingNeuralNetwork node must have at least two model heads to use sync node and outputs property. The model has {self._getModelHeadsLen()} heads."
            )
        return self._internal_sync.out

    @property
    def passthrough(self) -> dai.Node.Output:
        """Return the primary passthrough stream from the underlying NN node."""
        return self._nn.passthrough

    @property
    def passthroughs(self) -> dai.Node.OutputMap:
        """Return the passthrough output map from the underlying NN node."""
        return self._nn.passthroughs

    def getNumInferenceThreads(self) -> int:
        """Return the configured number of inference threads."""
        return self._nn.getNumInferenceThreads()

    @overload
    def getParser(self, index: int = 0) -> BaseParser | dai.DeviceNode:
        """Return the parser for the requested model head."""
        ...

    @overload
    def getParser(self, parserType: type[TParser], index: int = 0) -> TParser:
        """Return the parser for the requested model head."""
        ...

    def getParser(self, *args, **kwargs) -> BaseParser | dai.DeviceNode:
        """Return the parser for the requested model head.

        @param parserType: Optional expected parser type used for runtime type checking.
        @type parserType: type[TParser]
        @param index: Model head index. Defaults to 0.
        @type index: int
        @return: Parser node matching the requested head.
        @rtype: BaseParser | dai.DeviceNode
        """
        index = 0
        parser_type = None

        # Parse arguments based on the overload patterns
        if len(args) == 1:
            if isinstance(args[0], type):  # Case: getParser(parserType)
                parser_type = args[0]
            else:  # Case: getParser(index)
                index = args[0]
        elif len(args) == 2:  # Case: getParser(parser_type, index)
            parser_type = args[0]
            index = args[1]

        # Handle kwargs
        if "index" in kwargs:
            index = kwargs["index"]
        if "parserType" in kwargs:
            parser_type = kwargs["parserType"]

        if index not in self._parsers:
            raise KeyError(
                f"Parser with ID {index} not found. Available parser IDs: {list(self._parsers.keys())}"
            )
        parser = self._parsers[index]

        if parser_type:
            if not isinstance(parser, parser_type):
                raise TypeError(
                    f"Parser with ID {index} is of type: {type(parser)}. Requested type: {parser_type}"
                )
        return parser

    def getOutput(self, head: int) -> dai.Node.Output:
        """Return the output stream for the specified model head."""
        if head not in self._parsers:
            raise KeyError(
                f"Head {head} is not available. Available heads for the model {self._getModelName()} are {list(self._parsers.keys())}."
            )
        return self._parsers[head].out

    def setBackend(self, setBackend: str) -> None:
        """Set the backend used by the underlying neural-network node."""
        self._nn.setBackend(setBackend)

    def setBackendProperties(self, setBackendProperties: dict[str, str]) -> None:
        """Set backend-specific properties on the underlying neural-network node."""
        self._nn.setBackendProperties(setBackendProperties)

    def setBlob(self, blob: Path | dai.OpenVINO.Blob) -> None:
        """Set the blob used by the underlying neural-network node."""
        self._nn.setBlob(blob)

    def setBlobPath(self, path: Path) -> None:
        """Set the blob path used by the underlying neural-network node."""
        self._nn.setBlobPath(path)

    def setFromModelZoo(
        self, description: dai.NNModelDescription, useCached: bool
    ) -> None:
        """Load the model from the model zoo into the underlying NN node."""
        self._nn.setFromModelZoo(description, useCached)

    def setModelPath(self, modelPath: Path) -> None:
        """Set the model path used by the underlying neural-network node."""
        self._nn.setModelPath(modelPath)

    def setNNArchive(
        self, nnArchive: dai.NNArchive, numShaves: int | None = None
    ) -> None:
        """Set the active model archive and rebuild parser nodes.

        @param nnArchive: Neural-network archive containing the model and parser config.
        @type nnArchive: dai.NNArchive
        @param numShaves: Optional number of shaves allocated to the neural-network
            node.
        @type numShaves: int | None
        """
        self._nn_archive = nnArchive
        if numShaves:
            self._nn.setNNArchive(nnArchive, numShaves)  # type: ignore
        else:
            self._nn.setNNArchive(nnArchive)
        self._updateParsers(nnArchive)

    def setNumInferenceThreads(self, numThreads: int) -> None:
        """Sets the number of inference threads of the NeuralNetwork node."""
        self._nn.setNumInferenceThreads(numThreads)

    def setNumNCEPerInferenceThread(self, numNCEPerThread: int) -> None:
        """Sets the number of NCE per inference thread of the NeuralNetwork node."""
        self._nn.setNumNCEPerInferenceThread(numNCEPerThread)

    def setNumPoolFrames(self, numFrames: int) -> None:
        """Sets the number of pool frames of the NeuralNetwork node."""
        self._nn.setNumPoolFrames(numFrames)

    def setNumShavesPerInferenceThread(self, numShavesPerInferenceThread: int) -> None:
        """Sets the number of shaves per inference thread of the NeuralNetwork node."""
        self._nn.setNumShavesPerInferenceThread(numShavesPerInferenceThread)

    def build(
        self,
        input: dai.Node.Output | dai.node.Camera,
        nnSource: dai.NNModelDescription | dai.NNArchive | str,
        fps: float | None = None,
    ) -> "ParsingNeuralNetwork":
        """Build the neural-network node and create parser nodes for each head.

        @param input: Upstream output or camera feeding frames into the neural-network
            node.
        @type input: dai.Node.Output | dai.node.Camera
        @param nnSource: dai.NNModelDescription, dai.NNArchive, or HubAI model slug.
        @type nnSource: dai.NNModelDescription | dai.NNArchive | str
        @param fps: Optional runtime FPS limit for the neural-network node.
        @type fps: float | None
        @return: The configured node instance.
        @rtype: ParsingNeuralNetwork
        """

        platform = self.getParentPipeline().getDefaultDevice().getPlatformAsString()

        if isinstance(nnSource, str):
            nnSource = dai.NNModelDescription(nnSource)
        if isinstance(nnSource, (dai.NNModelDescription, str)):
            if not nnSource.platform:
                nnSource.platform = platform
            self._nn_archive = dai.NNArchive(dai.getModelFromZoo(nnSource))
        elif isinstance(nnSource, dai.NNArchive):
            self._nn_archive = nnSource
        else:
            raise ValueError(
                "nnSource must be either a NNModelDescription, NNArchive, or a string representing HubAI model slug."
            )

        kwargs = {"fps": fps} if fps else {}
        built_via_set_nn_archive = False

        try:
            self._nn.build(input, self._nn_archive, **kwargs)
        except RuntimeError as exc:
            if not self._should_use_manual_input_fallback(exc):
                raise

            self._logger.warning(
                "NNArchive build rejected grayscale image input. "
                "Falling back to explicit BGR preprocessing before the "
                "NeuralNetwork node."
            )
            prepared_input = self._prepare_manual_nn_input(input, fps=fps)
            self.setNNArchive(self._nn_archive)
            prepared_input.link(self._nn.input)
            built_via_set_nn_archive = True

        if not built_via_set_nn_archive:
            self._updateParsers(self._nn_archive)

        if len(self._parsers) > 1:
            self._createSyncNode()

        self._logger.debug(
            f"ParsingNeuralNetwork built with fps={fps}, type_of_nn_source={type(nnSource).__name__}, parsers={len(self._parsers)}"
        )

        return self

    def _should_use_manual_input_fallback(self, exc: RuntimeError) -> bool:
        message = str(exc)
        if "Unsupported input type" not in message:
            return False

        dai_type = self._get_archive_input_dai_type()
        return dai_type.startswith("GRAY")

    def _get_archive_input_dai_type(self) -> str:
        model_inputs = getattr(self._nn_archive.getConfig().model, "inputs", [])
        if not model_inputs:
            return ""

        preprocessing = getattr(model_inputs[0], "preprocessing", None)
        if preprocessing is None:
            return ""

        dai_type = getattr(preprocessing, "daiType", None) or getattr(
            preprocessing,
            "dai_type",
            None,
        )
        if dai_type is None:
            return ""
        if not isinstance(dai_type, str):
            dai_type = getattr(dai_type, "name", None) or getattr(
                dai_type,
                "value",
                "",
            )
        return str(dai_type).upper()

    def _prepare_manual_nn_input(
        self,
        input: dai.Node.Output | dai.node.Camera,
        *,
        fps: float | None,
    ) -> dai.Node.Output:
        nn_w = self._nn_archive.getInputWidth()
        nn_h = self._nn_archive.getInputHeight()
        platform = self.getParentPipeline().getDefaultDevice().getPlatform()
        frame_type = (
            dai.ImgFrame.Type.BGR888p
            if platform == dai.Platform.RVC2
            else dai.ImgFrame.Type.BGR888i
        )

        if isinstance(input, dai.node.Camera):
            request_kwargs = {
                "size": (nn_w, nn_h),
                "type": frame_type,
            }
            if fps is not None:
                request_kwargs["fps"] = fps
            return input.requestOutput(**request_kwargs)

        manip = self.getParentPipeline().create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(nn_w * nn_h * 3)
        manip.initialConfig.setFrameType(frame_type)
        manip.initialConfig.setOutputSize(
            w=nn_w,
            h=nn_h,
            mode=dai.ImageManipConfig.ResizeMode.CENTER_CROP,
        )
        input.link(manip.inputImage)
        self._input_adapter = manip
        return manip.out

    def run(self) -> None:
        """Methods inherited from ThreadedHostNode.

        Method runs with start of the pipeline.
        """
        self._checkNNArchive()

    def cleanup(self):
        """Cleans up the ParsingNeuralNetwork node and removes all nodes created by
        ParsingNeuralNetwork from the pipeline.

        Must be called before removing the node from the pipeline.
        """
        self.getParentPipeline().remove(self._nn)
        self._nn = None

        self._removeOldParserNodes()
        if self._input_adapter is not None:
            self.getParentPipeline().remove(self._input_adapter)
            self._input_adapter = None
        self._internal_sync = None
        self._parsers = {}

    def _updateParsers(self, nnArchive: dai.NNArchive) -> None:
        self._removeOldParserNodes()
        self._parsers = self._getParserNodes(nnArchive)

    def _removeOldParserNodes(self) -> None:
        pipeline = self.getParentPipeline()
        for parser in self._parsers.values():
            pipeline.remove(parser)
        if self._internal_sync is not None:
            pipeline.remove(self._internal_sync)

    def _getParserNodes(self, nnArchive: dai.NNArchive) -> dict[int, BaseParser]:
        parser_generator = self.getParentPipeline().create(ParserGenerator)
        parsers = self._generateParsers(parser_generator, nnArchive)
        for parser in parsers.values():
            self._nn.out.link(
                parser.input
            )  # TODO: once NN node has output dictionary, link to the correct output
        self.getParentPipeline().remove(parser_generator)
        return parsers

    def _generateParsers(
        self, parserGenerator: ParserGenerator, nnArchive: dai.NNArchive
    ) -> dict[int, BaseParser]:
        return parserGenerator.build(nnArchive)

    def _getModelHeadsLen(self):
        heads = self._getModelHeads()
        if not heads:
            return 0
        return len(heads)

    def _getModelHeads(self):
        return self._getConfig().model.heads

    def _checkNNArchive(self) -> None:
        if self._nn_archive is None:
            raise RuntimeError(
                f"NNArchive is not set. Use {self.setNNArchive.__name__} or {self.build.__name__} method to set it."
            )

    def _getModelName(self) -> str:
        return self._getConfig().model.metadata.name

    def _getConfig(self):
        return self._nn_archive.getConfig()

    def _createSyncNode(self):
        if self._internal_sync is not None:
            raise RuntimeError(
                "Sync node already exists. Remove it before creating a new one."
            )
        if len(list(self._parsers.values())) <= 1:
            raise RuntimeError(
                "ParsingNeuralNetwork node must have at least two model heads to use sync node."
            )
        self._internal_sync = self.getParentPipeline().create(dai.node.Sync)
        self._internal_sync.setRunOnHost(True)
        outputs = [parser.out for parser in list(self._parsers.values())]
        for ix, output in enumerate(outputs):
            output.link(self._internal_sync.inputs[str(ix)])
        self._logger.debug(f"Internal Sync node created with {len(outputs)} outputs")
