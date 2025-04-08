from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork


class HostParsingNeuralNetwork(ParsingNeuralNetwork):
    def _generateParsers(self, parserGenerator, nnArchive):
        return parserGenerator.build(nnArchive, host_only=True)
