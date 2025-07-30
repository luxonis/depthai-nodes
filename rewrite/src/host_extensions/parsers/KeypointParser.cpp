//
// Created by thwdpc on 7/25/25.
//

#include "parsers/KeypointParser.hpp"

#include <spdlog/spdlog.h>

#include "ErrorMacros.hpp"

namespace dai::node {
void KeypointParser::buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) {
    bool fallback = false;
    if(const auto layers = head.metadata.keypointsOutputs) {
        for(auto& layerName : *layers) {
            auto output = std::find_if(model.outputs.begin(), model.outputs.end(), [&](const auto& o) { return o.name == layerName; });
            DAI_CHECK_V(output != model.outputs.end(), "{}: keypoint output {} not found in model", NAME, layerName);
            keypointsOutputs.push_back(*output);
        }
    } else {
        // TODO built in logging utils seem to be unavailable?
        spdlog::trace("KeypointParser(or subclass) did not receive keypoints_outputs, fallback to using all outputs");
        for(auto& output : model.outputs) {
            keypointsOutputs.push_back(output);
        };
        fallback = true;
    }

    const uint8_t ko_sz = keypointsOutputs.size();
    if(ko_sz < 1 || ko_sz > 3) {
        std::string where = fallback ? "During fallback to use all outputs" : "Configured keypoints_outputs";
        throw std::runtime_error(fmt::format("{w}: size {sz} must satisfy 1 <= {sz} <= 3 ", fmt::arg("r", where), fmt::arg("sz", ko_sz)));
    }

    // take outputs size if it makes sense else default
    valuesPerKeypoint = head.metadata.extraParams.value("values_per_keypoint", ko_sz > 1 ? ko_sz : valuesPerKeypoint);

    DAI_CHECK_V(ko_sz == 1 || ko_sz == valuesPerKeypoint,
                "Expected one output per keypoint dimension, or one output that contains all keypoints, got {} layers vs dimensionality {}.",
                ko_sz,
                nKeypoints);

    if(const auto n = head.metadata.nKeypoints) {
        nKeypoints = *n;
    } else {
        spdlog::warn("SimCCKeypointParser did not receive n_keypoints, defaulting to standard COCO 17. Populating this field is strongly encouraged");
    }

    keypointNames = head.metadata.extraParams.value("keypoint_names", keypointNames);
    skeletonEdges = head.metadata.extraParams.value("skeleton_edges", skeletonEdges);
}
void KeypointParser::run() {
    UNIMPLEMENTED_V("KeypointParser::run")
}

}  // namespace dai::node