//
// Created by thwdpc on 7/24/25.
//

#include "ParsingNeuralNetwork.hpp"

#include <iosfwd>
#include <utility>
#include <variant>
#include <vector>
namespace dai::node {

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(Output& input, const NNArchive& nnArchive) {
    nn->build(input, nnArchive);
    updateParsers(nnArchive);
    return std::static_pointer_cast<ParsingNeuralNetwork>(shared_from_this());
}

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(const std::shared_ptr<Camera>& input,
                                                                  NNModelDescription modelDesc,
                                                                  std::optional<float> fps) {
    nn->build(input, std::move(modelDesc), fps);
    try {
        const NNArchive& archive = nn->getNNArchive().value();
        updateParsers(archive);
    } catch(std::bad_optional_access& e) {
        std::cout << "NeuralNetwork node failed to create an archive and failed silently, getNNArchive returned std::nullopt: " << e.what() << std::endl;
    }
    return std::static_pointer_cast<ParsingNeuralNetwork>(shared_from_this());
}

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(const std::shared_ptr<Camera>& input, const NNArchive& nnArchive, std::optional<float> fps) {
    nn->build(input, nnArchive, fps);
    updateParsers(nnArchive);
    return std::static_pointer_cast<ParsingNeuralNetwork>(shared_from_this());
}

// Updates parsers based on the provided NNArchive
void ParsingNeuralNetwork::updateParsers(const NNArchive& nnArchive) {
    removeOldParserNodes();
    parsers = getParserNodes(nnArchive);
}

// Removes previously created parser nodes and internal sync node from pipeline
void ParsingNeuralNetwork::removeOldParserNodes() {
    for(const auto& entry : parsers) {
        if(std::holds_alternative<std::shared_ptr<BaseParser>>(entry)) {
            auto parser = std::get<std::shared_ptr<DetectionParser>>(entry);
            getParentPipeline().remove(parser);
        } else if(std::holds_alternative<std::shared_ptr<DetectionParser>>(entry)) {
            auto parser = std::get<std::shared_ptr<DetectionParser>>(entry);
            getParentPipeline().remove(parser);
        }
    }
    parsers.clear();
    inputs.clear();
}

// Creates new parser nodes from NNArchive, links their input/output, and returns them
std::vector<HostOrDeviceParser> ParsingNeuralNetwork::getParserNodes(const NNArchive& nnArchive) {
    auto newParsers = ParserGenerator::generateAllParsers(getParentPipeline(), nnArchive);

    for(std::size_t idx = 0; idx < newParsers.size(); ++idx) {
        const auto& entry = newParsers[idx];
        // Link NN output to parser input (adjust as needed for output types)
        if(std::holds_alternative<std::shared_ptr<BaseParser>>(entry)) {
            auto parser = std::get<std::shared_ptr<DetectionParser>>(entry);
            nn->out.link(parser->input);
            parser->out.link(inputs[std::to_string(idx)]);
        } else if(std::holds_alternative<std::shared_ptr<DetectionParser>>(entry)) {
            auto parser = std::get<std::shared_ptr<DetectionParser>>(entry);
            nn->out.link(parser->input);
            parser->out.link(inputs[std::to_string(idx)]);
        } else {
            throw std::runtime_error("Parser type not supported.");
        }
    }

    return newParsers;
}

}  // namespace dai::node