//
// Created by thwdpc on 7/24/25.
//

#include "ParserGenerator.hpp"

#include <iosfwd>
#include <variant>
#include <vector>

namespace dai::node {

// Utility for device parser names
const std::vector<std::string> DEVICE_PARSERS = {"YOLO", "SSD"};

static const std::unordered_map<std::string, std::function<std::shared_ptr<BaseParser>()>> parserMap = {
    // { "YOLOExtendedParser", [](){ return std::make_shared<YOLOExtendedParser>(); } },
    { "SimCCKeypointParser", [](){ return std::make_shared<SimCCKeypointParser>(); } }
};


std::string getHostParserName(const std::string& parserName) {
    if(parserName == "YOLO") {
        return "YOLOExtendedParser";
    }
    throw std::runtime_error("Parser " + parserName + " is not supported for host only mode.");
}

std::vector<HostOrDeviceParser> ParserGenerator::generateAllParsers(Pipeline pipeline, const NNArchive& nnArchive, bool hostOnly) {
    auto platform = pipeline.getDefaultDevice()->getPlatform();

    auto info = archiveGetModelEnsureOneHeadV1(nnArchive, platform);

    std::vector<HostOrDeviceParser> parsers;

    for(int i = 0; i < info.heads.size(); i++) {
        HostOrDeviceParser parser = generateOneV1Parser(pipeline, nnArchive, info.heads[i], info.model, hostOnly);
        parsers.push_back(std::move(parser));
    }
}

HostOrDeviceParser ParserGenerator::generateParser(Pipeline& pipeline, const NNArchive& nnArchive, int headIndex, bool hostOnly) {
    auto platform = pipeline.getDefaultDevice()->getPlatform();
    auto info = archiveGetModelEnsureOneHeadV1(nnArchive, platform);

    return generateOneV1Parser(pipeline, nnArchive, info.heads[headIndex], info.model, hostOnly);
}

ConfigModelWithHeads ParserGenerator::archiveGetModelEnsureOneHeadV1(const NNArchive& nnArchive, Platform targetPlatform) {
    const auto& nnArchiveCfg = nnArchive.getVersionedConfig();

    DAI_CHECK_V(nnArchiveCfg.getVersion() == NNArchiveConfigVersion::V1, "Only V1 configs are supported for NeuralNetwork.build method");
    auto supportedPlatforms = nnArchive.getSupportedPlatforms();
    bool platformSupported = std::find(supportedPlatforms.begin(), supportedPlatforms.end(), targetPlatform) != supportedPlatforms.end();
    DAI_CHECK_V(platformSupported, "Platform not supported by the neural network model");

    // Get model heads
    auto config = nnArchive.getConfig<nn_archive::v1::Config>();

    if(auto headsOpt = config.model.heads) {
        auto headsV1 = *headsOpt;
        if(!headsV1.empty()) {
            return ConfigModelWithHeads{.model = config.model, .heads = headsV1};
        }
    }
    throw std::runtime_error(fmt::format("No heads defined in the NN Archive."));
}

HostOrDeviceParser ParserGenerator::generateOneV1Parser(
    Pipeline& pipeline, const NNArchive& owningArchive, const nn_archive::v1::Head& head, const nn_archive::v1::Model model, bool hostOnly) {
    std::string parser_name = head.parser;
    bool is_device_parser = std::find(DEVICE_PARSERS.begin(), DEVICE_PARSERS.end(), parser_name) != DEVICE_PARSERS.end();

    if(is_device_parser) {
        if(hostOnly) {
            parser_name = getHostParserName(parser_name);
        } else {
            // Device parser handling
            auto device_parser = pipeline.create<dai::node::DetectionParser>();
            device_parser->setNNArchive(owningArchive);
            return device_parser;
        }
    }

    // Create parser via factory
    auto creator = parserMap.find(parser_name);
    auto parser = creator->second()->build(head, model);
    return parser;
}

}  // namespace dai::node