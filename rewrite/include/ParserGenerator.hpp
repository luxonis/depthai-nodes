//
// Created by thwdpc on 7/24/25.
//

#pragma once
#include <variant>

#include "ErrorMacros.hpp"
#include "depthai/depthai.hpp"
#include "parsers/BaseParser.hpp"

namespace dai::node {

typedef std::variant<std::shared_ptr<BaseParser>, std::shared_ptr<DetectionParser>> HostOrDeviceParser;

struct ConfigModelWithHeads {
    nn_archive::v1::Model model;
    std::vector<nn_archive::v1::Head> heads;
};

class ParserGenerator {
   public:
    static std::vector<HostOrDeviceParser> generateAllParsers(Pipeline pipeline, const NNArchive& nnArchive, bool hostOnly = false);
    static HostOrDeviceParser generateParser(Pipeline& pipeline, const NNArchive& nnArchive, int headIndex, bool hostOnly = false);

   private:
    static ConfigModelWithHeads archiveGetModelEnsureOneHeadV1(const NNArchive& nnArchive, Platform targetPlatform);
    static HostOrDeviceParser generateOneV1Parser(
        Pipeline& pipeline, const NNArchive& owningArchive, const nn_archive::v1::Head& head, const nn_archive::v1::Model& model, bool hostOnly = false);
};
}  // namespace dai::node
