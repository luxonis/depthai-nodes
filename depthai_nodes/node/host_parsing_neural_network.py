from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork


class HostParsingNeuralNetwork(ParsingNeuralNetwork):
    """Run inference with the host parsers implemented by depthai-nodes.

    Uses the same build arguments and ports as ``ParsingNeuralNetwork``,
    while selecting ``ParserGenerator.build(hostOnly=True)`` for each model
    head. Use this node when the Python parser implementation is required.
    """

    def _generateParsers(self, parserGenerator, nnArchive):
        return parserGenerator.build(nnArchive, hostOnly=True)
