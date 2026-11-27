# RGB axial-depth correction

`ImageAlignDepthCorrection` converts aligned source-camera axial Z into
target-camera axial Z before native PointCloud generation. The verified SDK is
DepthAI 3.12.0. ImageAlign places source depth values on a target pixel grid;
the correction accounts for the cameras' different optical axes and origins.

The [node class documentation](../depthai_nodes/node/image_align_depth_correction.py)
is the reference for wiring, explicit policies, stream association, calibration,
outputs and provenance handling. The
[array-correction API](../depthai_nodes/utils/axial_depth.py) accepts explicit
geometry when native frames are unavailable. Keep original RGB and stereo views
as separate detection inputs.

## Conversion geometry

For `T_source_target = [R | t]` and integer target pixel ray
`r = inverse(K_target) [u, v, 1]`, the conversion is
`Z_target = (Z_source - t_z) / (R[2] dot r)`.

The target grid must be undistorted and perspective. Source intrinsics determine
alignment sampling; the actual source pose, including rectification, and target
intrinsics determine the axial conversion. A camera's native pose cannot replace
its rectified depth pose.

Arithmetic is float64 with nearest-integer millimeter rounding and ties to even.
Source quantization contributes up to `0.5 / denominator` mm of target error,
in addition to output rounding. Depths 0 and 65535 remain invalid sentinels;
newly invalid conversions become zero. Consumers must retain validity masks and
point order when constructing or filtering clouds.

Correction is an upstream pipeline step. Existing box nodes and the frozen
MeasureAnything implementation consume the resulting cloud without repeating it.

## Verification and limits

The [pipeline example](../examples/image_align_depth_correction/README.md)
reproduces synthetic ImageAlign and PointCloud checks on the host or in an OAK-4
standalone app. Recorded checks on November 24, 2026 used DepthAI 3.12.0 and
matched corrected depth bytes on both paths. With 3,331 valid pixels, maximum
corrected source-plane residual was 0.530 mm, versus 84.235 mm without correction.
These results describe that fixture, not general live NeuralDepth accuracy or
performance.

Production integration still needs checks of real NeuralDepth 768x480
rectification metadata, calibration changes and replay across devices. Reference
scene comparisons must verify cloud count, point order, validity, image
association and unchanged measurement outcomes before replacing another
correction. The synthetic checks do not establish those properties or live
throughput.
