//
// Created by thwdpc on 7/24/25.
//

#include "ParserGenerator.hpp"

#include <iosfwd>
#include <variant>
#include <vector>

#include "parsers/SimCCKeypointParser.hpp"

namespace dai::node {

// Utility for device parser names
const std::vector<std::string> DEVICE_PARSERS = {"YOLO", "SSD"};

static const std::unordered_map<std::string, std::function<std::shared_ptr<BaseParser>()>> parserMap = {
    // { "YOLOExtendedParser", [](){ return std::make_shared<YOLOExtendedParser>(); } },
    { "SimCCKeypointParser", [](){ return std::make_shared<SimCCKeypointParser>(); } }
};


std::string getHostParserName(const std::string& parserName) {
    TODO_V("MobileNet");
    if(parserName == "YOLO") {
        return "YOLOExtendedParser";
    }
    throw std::runtime_error("Parser " + parserName + " is not supported for host only mode.");
}

std::vector<HostOrDeviceParser> ParserGenerator::generateAllParsers(Pipeline pipeline, const NNArchive& nnArchive, const bool hostOnly) {
    const auto platform = pipeline.getDefaultDevice()->getPlatform();

    auto [model, heads] = archiveGetModelEnsureOneHeadV1(nnArchive, platform);

    std::vector<HostOrDeviceParser> parsers;

    for(int i = 0; i < heads.size(); i++) {
        HostOrDeviceParser parser = generateOneV1Parser(pipeline, nnArchive, heads[i], model, hostOnly);
        parsers.push_back(std::move(parser));
    }
    return parsers;
}

HostOrDeviceParser ParserGenerator::generateParser(Pipeline& pipeline, const NNArchive& nnArchive, const int headIndex, const bool hostOnly) {
    auto platform = pipeline.getDefaultDevice()->getPlatform();
    auto info = archiveGetModelEnsureOneHeadV1(nnArchive, platform);

    return generateOneV1Parser(pipeline, nnArchive, info.heads[headIndex], info.model, hostOnly);
}

ConfigModelWithHeads ParserGenerator::archiveGetModelEnsureOneHeadV1(const NNArchive& nnArchive, const Platform targetPlatform) {
    const auto& nnArchiveCfg = nnArchive.getVersionedConfig();

    DAI_CHECK_V(nnArchiveCfg.getVersion() == NNArchiveConfigVersion::V1, "Only V1 configs are supported for NeuralNetwork.build method");
    auto supportedPlatforms = nnArchive.getSupportedPlatforms();
    bool platformSupported = std::ranges::find(supportedPlatforms, targetPlatform) != supportedPlatforms.end();
    DAI_CHECK_V(platformSupported, "Platform not supported by the neural network model");

    // Get model heads
    auto config = nnArchive.getConfig<nn_archive::v1::Config>();

    if(const auto headsOpt = config.model.heads) {
        if(const auto headsV1 = *headsOpt; !headsV1.empty()) {
            return ConfigModelWithHeads{.model = config.model, .heads = headsV1};
        }
    }
    throw std::runtime_error(fmt::format("No heads defined in the NN Archive."));
}

HostOrDeviceParser ParserGenerator::generateOneV1Parser(
    Pipeline& pipeline, const NNArchive& owningArchive, const nn_archive::v1::Head& head, const nn_archive::v1::Model& model, const bool hostOnly) {
    std::string parser_name = head.parser;

    // If this *could* be an on-device parser(currently just DetectionParser) then check whether that's allowed by !hostOnly
    if(std::ranges::find(DEVICE_PARSERS, parser_name) != DEVICE_PARSERS.end() && !hostOnly) {
        // Device parser handling
        auto device_parser = pipeline.create<DetectionParser>();
        device_parser->setNNArchive(owningArchive);
        return device_parser;
    }
    parser_name = getHostParserName(parser_name);
    DAI_CHECK(parserMap.contains(parser_name), "Parser " + parser_name + " not found");
    auto parser = parserMap.find(parser_name)->second()->build(head, model);
    return parser;
}

}  // namespace dai::node