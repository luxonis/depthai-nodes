from pathlib import Path

import depthai as dai


class ParsingNeuralNetwork(dai.node.ThreadedHostNode):
    Propeties = dai.node.NeuralNetwork.Properties

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._nn = self.getParentPipeline().create(dai.node.NeuralNetwork)
        self._nn.out.createOutputQueue().addCallback(self.forwardParsedOutput)
        self._output = self.createOutput()

    def build(
        self, input: dai.Node.Output, nn_archive: dai.NNArchive
    ) -> "ParsingNeuralNetwork":
        self._nn.build(input, nn_archive)
        # TODO: create parser accordingly to nn_archive
        # self._parser = self.getParentPipeline().create(dai.node.)
        return self

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
        self._nn.setNNArchive(nnArchive)

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
        return self._output

    @property
    def passthrough(self) -> dai.Node.Output:
        return self._nn.passthrough

    @property
    def passthroughs(self) -> dai.Node.OutputMap:
        return self._nn.passthroughs

    def run(self) -> None:
        # ThreadedHostNode will break if the run method is not defined
        pass

    def forwardParsedOutput(self, nn_data: dai.NNData):
        self._output.send(nn_data)  # TEST
        # self._output.send(parsed_output)
