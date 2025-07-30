//
// Created by thwdpc on 7/28/25.
//

#include "messages/Keypoints.hpp"

#include <spdlog/spdlog.h>

#include "ErrorMacros.hpp"
namespace dai {
template <typename X, typename Y, typename Z>
Keypoints<X, Y, Z>::Keypoints(std::shared_ptr<NNData>&& other, xt::xarray<float>&& planarStackedKeypoints) {
    // KP#, dim
    const size_t numKeypoints = planarStackedKeypoints.shape()[0], numDimsFound = planarStackedKeypoints.shape()[1];
    // I don't love this but
    const uint8_t x = !std::is_void_v<X> ? sizeof(X) / sizeof(float) : 0, y = !std::is_void_v<Y> ? sizeof(Y) / sizeof(float) : 0, z = !std::is_void_v<Z> ? sizeof(Z) / sizeof(float) : 0;
    const uint8_t numDimsFromType = x + y + z;

    DAI_CHECK_V(numDimsFound == numDimsFromType,
                "Trying to build {} dimensional keypoints, got {} sets of keypoints/confidence values",
                numDimsFromType,
                numDimsFound);

    kpVec = std::vector<Keypoint<X, Y, Z>>(numKeypoints);
    // Direct copy into the vec
    assert(sizeof(Keypoint<X, Y, Z>) == sizeof(float) * numDimsFromType);
    assert(planarStackedKeypoints.size() == numKeypoints * numDimsFromType);
    std::memcpy(kpVec.data(), planarStackedKeypoints.data(), planarStackedKeypoints.size() * sizeof(float));

    transformation = other->transformation;
    setTimestamp(other->getTimestamp());
    setSequenceNum(other->sequenceNum);
}
}  // namespace dai