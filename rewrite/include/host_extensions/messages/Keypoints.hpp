//
// Created by thwdpc on 7/28/25.
//

#pragma once
#include <depthai/common/ImgTransformations.hpp>
#include <depthai/pipeline/datatype/Buffer.hpp>
#include <depthai/pipeline/datatype/NNData.hpp>

namespace dai {

struct ValueWithConfidence {
    float_t value;
    float_t confidence;
};

template<typename X, typename Y, typename Z>
struct Keypoint {
    X x;
    Y y;
    Z z;
};

template<typename X, typename Y>
struct Keypoint<X, Y, void> {
    X x;
    Y y;
};


template <typename X = ValueWithConfidence, typename Y = ValueWithConfidence, typename Z = void>
class Keypoints : public Buffer {
   public:
    std::optional<ImgTransformation> transformation;

    std::vector<Keypoint<X, Y, Z>> kpVec;

    Keypoints(std::shared_ptr<NNData>&& other, xt::xarray<float>&& planarStackedKeypoints);
};

typedef Keypoints<ValueWithConfidence, ValueWithConfidence, ValueWithConfidence> Keypoints3D;
typedef Keypoints<float_t, ValueWithConfidence> Keypoints2D;
typedef Keypoints<ValueWithConfidence, ValueWithConfidence> Keypoints2D2V;

}  // namespace dai