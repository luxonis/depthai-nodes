//
// Created by thwdpc on 7/24/25.
//
#pragma once
#include <iosfwd>
#include <variant>
#include <vector>

#include "ParserGenerator.hpp"
#include "depthai/depthai.hpp"
#include "parsers/BaseParser.hpp"

namespace dai::node {

class ParsingNeuralNetwork : public CustomNode<ParsingNeuralNetwork> {
    std::shared_ptr<ParsingNeuralNetwork> build(Output& input, const NNArchive& nnArchive);
    std::shared_ptr<ParsingNeuralNetwork> build(const std::shared_ptr<Camera>& input, NNModelDescription modelDesc, std::optional<float> fps = std::nullopt);
    std::shared_ptr<ParsingNeuralNetwork> build(const std::shared_ptr<Camera>& input, const NNArchive& nnArchive, std::optional<float> fps = std::nullopt);

   private:
    std::vector<HostOrDeviceParser> getParserNodes(const NNArchive& nnArchive);

    void updateParsers(const NNArchive& nnArchive);

    void removeOldParserNodes();
    Subnode<NeuralNetwork> nn{*this, "nn"};

   protected:
    std::vector<HostOrDeviceParser> parsers;

};

}  // namespace dai::node
