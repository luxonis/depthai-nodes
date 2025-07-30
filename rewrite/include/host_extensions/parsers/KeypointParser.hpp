//
// Created by thwdpc on 7/25/25.
//

#pragma once
#include "BaseParser.hpp"

namespace dai::node {
class KeypointParser : public BaseParser {
public:
    const char* getName() const override{ return "KeypointParser"; };

protected:
    void buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) override;
    void run() override;

    std::vector<nn_archive::v1::Output> keypointsOutputs{};
    uint16_t nKeypoints = 17;
    // dimensionality: 2D or 3D
    uint8_t valuesPerKeypoint = 2;
    std::vector<std::string> keypointNames{};
    std::vector<std::pair<uint8_t, uint8_t>> skeletonEdges{};
};
}