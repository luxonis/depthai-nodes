from pathlib import Path

import depthai as dai

from .ml.parsers import BaseParser, ParserGenerator


class ParsingNeuralNetwork(dai.node.ThreadedHostNode):
    Propeties = dai.node.NeuralNetwork.Properties

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pipeline = self.getParentPipeline()
        self._nn = self._pipeline.create(dai.node.NeuralNetwork)
        self._parsers: dict[int, BaseParser] = {}

    def build(
        self, input: dai.Node.Output, nnArchive: dai.NNArchive
    ) -> "ParsingNeuralNetwork":
        self._nn_archive = nnArchive
        self._nn.build(input, nnArchive)
        self._updateParsers(nnArchive)
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
            self._nn.out.link(parser.input)
        self._pipeline.remove(parser_generator)
        return parsers

    def getNumInferenceThreads(self) -> int:
        return self._nn.getNumInferenceThreads()

    def setBackend(self, setBackend: str) -> None:
        self._nn.setBackend(setBackend)

    def setBackendProperties(self, setBackendProperties: dict[str, str]) -> None:
        self._nn.setBackendProperties(setBackendProperties)

    def setBlob(self, blob: Path | dai.OpenVINO.Blob) -> None:
        self._nn.setBlob(blob)

    def setBlobPath(self, path: Path) -> None:
        self._nn.setBlobPath(path)

    def setFromModelZoo(
        self, description: dai.NNModelDescription, useCached: bool
    ) -> None:
        self._nn.setFromModelZoo(description, useCached)

    def setModelPath(self, modelPath: Path) -> None:
        self._nn.setModelPath(modelPath)

    def setNNArchive(self, nnArchive: dai.NNArchive) -> None:
        self._nn_archive = nnArchive
        self._nn.setNNArchive(nnArchive)
        self._updateParsers(nnArchive)

    def setNumInferenceThreads(self, numThreads: int) -> None:
        self._nn.setNumInferenceThreads(numThreads)

    def setNumNCEPerInferenceThread(self, numNCEPerThread: int) -> None:
        self._nn.setNumNCEPerInferenceThread(numNCEPerThread)

    def setNumPoolFrames(self, numFrames: int) -> None:
        self._nn.setNumPoolFrames(numFrames)

    def setNumShavesPerInferenceThread(self, numShavesPerInferenceThread: int) -> None:
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
        return list(self._parsers.values())[0].output

    def _getModelHeads(self):
        return self._getConfig().model.heads

    @property
    def passthrough(self) -> dai.Node.Output:
        return self._nn.passthrough

    @property
    def passthroughs(self) -> dai.Node.OutputMap:
        return self._nn.passthroughs

    def run(self) -> None:
        self._checkNNArchive()

    def _checkNNArchive(self) -> None:
        if self._nn_archive is None:
            raise RuntimeError(
                f"NNArchive is not set. Use {self.setNNArchive.__name__} or {self.build.__name__} method to set it."
            )

    def getOutput(self, head: int) -> dai.Node.Output:
        if head not in self._parsers:
            raise KeyError(
                f"Head {head} is not available. Available heads for the model {self._getModelName()} are {list(self._parsers.keys())}."
            )
        return self._parsers[head].output

    def _getModelName(self) -> str:
        return self._getConfig().model.metadata.name

    def _getConfig(self):
        return self._nn_archive.getConfig().getConfigV1()
