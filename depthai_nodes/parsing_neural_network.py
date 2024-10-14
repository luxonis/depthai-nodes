from pathlib import Path
from typing import overload

import depthai as dai

from .ml.parsers import BaseParser
from .parser_generator import ParserGenerator


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
    passthrough : Node.Output
        Neural network passthrough.
    passthroughs : Node.OutputMap
        Neural network passthroughs.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the ParsingNeuralNetwork node. NeuralNetwork node is created in
        the pipeline.

        @param args: Arguments to be passed to the ThreadedHostNode class.
        @param kwargs: Keyword arguments to be passed to the ThreadedHostNode class.
        """
        super().__init__(*args, **kwargs)
        self._pipeline = self.getParentPipeline()
        self._nn = self._pipeline.create(dai.node.NeuralNetwork)
        self._parsers: dict[int, BaseParser] = {}

    @overload
    def method(
        self, input: dai.Node.Output, nn_source: dai.NNModelDescription, fps: int
    ) -> "ParsingNeuralNetwork":
        ...

    @overload
    def method(
        self, input: dai.Node.Output, nn_source: dai.NNArchive, fps: int
    ) -> "ParsingNeuralNetwork":
        ...

    def build(
        self,
        input: dai.Node.Output,
        nn_source,
        fps: int = None,
    ) -> "ParsingNeuralNetwork":
        """Builds the underlying NeuralNetwork node and creates parser nodes for each
        model head.

        Attributes
        ----------

        input : Node.Input
            Node's input. It is a linking point to which the NeuralNetwork is linked. It accepts the output of a Camera node.
        nn_source : Union[dai.NNModelDescription, dai.NNArchive]
            NNModelDescription object containing the HubAI model descriptors, or NNArchive object of the model.
        fps_limit : int
            FPS limit for the model runtime.

        Returns
        -------
        ParsingNeuralNetwork
            Returns the ParsingNeuralNetwork object.

        Raises
        ------
        ValueError
            If the nn_source is not a NNModelDescription or NNArchive object.
        """

        if isinstance(nn_source, dai.NNModelDescription):
            if not nn_source.platform:
                nn_source.platform = (
                    self.getParentPipeline().getDefaultDevice().getPlatformAsString()
                )
            self._nn_archive = dai.NNArchive(dai.getModelFromZoo(nn_source))
        elif isinstance(nn_source, dai.NNArchive):
            self._nn_archive = nn_source
        else:
            raise ValueError(
                "nn_source must be either a NNModelDescription or NNArchive"
            )

        kwargs = {"fps": fps} if fps else {}
        self._nn.build(input, nn_source, **kwargs)

        self._updateParsers(self._nn_archive)
        return self

    def _updateParsers(self, nnArchive: dai.NNArchive) -> None:
        self._removeOldParserNodes()
        self._parsers = self._getParserNodes(nnArchive)

    def _removeOldParserNodes(self) -> None:
        for parser in self._parsers.values():
            self._pipeline.remove(parser)

    def _getParserNodes(self, nnArchive: dai.NNArchive) -> dict[int, BaseParser]:
        parser_generator = self._pipeline.create(ParserGenerator)
        parsers = parser_generator.build(nnArchive)
        for parser in parsers.values():
            self._nn.out.link(
                parser.input
            )  # TODO: once NN node has output dictionary, link to the correct output
        self._pipeline.remove(parser_generator)
        return parsers

    def getNumInferenceThreads(self) -> int:
        """Returns number of inference threads of the NeuralNetwork node."""
        return self._nn.getNumInferenceThreads()

    def setBackend(self, setBackend: str) -> None:
        """Sets the backend of the NeuralNetwork node."""
        self._nn.setBackend(setBackend)

    def setBackendProperties(self, setBackendProperties: dict[str, str]) -> None:
        """Sets the backend properties of the NeuralNetwork node."""
        self._nn.setBackendProperties(setBackendProperties)

    def setBlob(self, blob: Path | dai.OpenVINO.Blob) -> None:
        """Sets the blob of the NeuralNetwork node."""
        self._nn.setBlob(blob)

    def setBlobPath(self, path: Path) -> None:
        """Sets the blob path of the NeuralNetwork node."""
        self._nn.setBlobPath(path)

    def setFromModelZoo(
        self, description: dai.NNModelDescription, useCached: bool
    ) -> None:
        """Sets the model from the model zoo of the NeuralNetwork node."""
        self._nn.setFromModelZoo(description, useCached)

    def setModelPath(self, modelPath: Path) -> None:
        """Sets the model path of the NeuralNetwork node."""
        self._nn.setModelPath(modelPath)

    def setNNArchive(self, nnArchive: dai.NNArchive) -> None:
        """Sets the NNArchive of the ParsingNeuralNetwork node.

        Updates the NeuralNetwork node and parser nodes.
        """
        self._nn_archive = nnArchive
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

    @property
    def input(self) -> dai.Node.Input:
        return self._nn.input

    @property
    def inputs(self) -> dai.Node.InputMap:
        return self._nn.inputs

    @property
    def out(self) -> dai.Node.Output:
        self._checkNNArchive()
        if len(self._parsers) != 1:
            raise RuntimeError(
                f"Property out is only available when there is exactly one model head. \
                               The model has {len(self._getModelHeads())} heads. Use {self.getOutput.__name__} method instead."
            )
        return list(self._parsers.values())[0].out

    def _getModelHeads(self):
        return self._getConfig().model.heads

    @property
    def passthrough(self) -> dai.Node.Output:
        return self._nn.passthrough

    @property
    def passthroughs(self) -> dai.Node.OutputMap:
        return self._nn.passthroughs

    def run(self) -> None:
        """Methods inherited from ThreadedHostNode.

        Method runs with start of the pipeline.
        """
        self._checkNNArchive()

    def _checkNNArchive(self) -> None:
        if self._nn_archive is None:
            raise RuntimeError(
                f"NNArchive is not set. Use {self.setNNArchive.__name__} or {self.build.__name__} method to set it."
            )

    def getOutput(self, head: int) -> dai.Node.Output:
        """Obtains output of a parser for specified NeuralNetwork model head."""
        if head not in self._parsers:
            raise KeyError(
                f"Head {head} is not available. Available heads for the model {self._getModelName()} are {list(self._parsers.keys())}."
            )
        return self._parsers[head].out

    def _getModelName(self) -> str:
        return self._getConfig().model.metadata.name

    def _getConfig(self):
        return self._nn_archive.getConfig().getConfigV1()
