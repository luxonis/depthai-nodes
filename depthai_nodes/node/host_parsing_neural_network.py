from depthai_nodes.node import ParsingNeuralNetwork


class HostParsingNeuralNetwork(ParsingNeuralNetwork):
    def _generateParsers(self, parserGenerator, nnArchive):
        return parserGenerator.build(nnArchive, host_only=True)
