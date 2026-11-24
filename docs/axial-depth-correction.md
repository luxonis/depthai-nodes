# RGB axial-depth correction

`ImageAlignDepthCorrection` converts aligned source-camera axial Z into
target-camera axial Z before native PointCloud generation. Values remain RAW16
millimeters. The initial SDK pin is `depthai==3.12.0`.

ImageAlign places samples on the target pixel grid while retaining source axial
Z. For the initial benchmark profile, the target is an undistorted RGB pinhole
grid. The source is the actual depth optical frame, including its rectification.
The original RGB and stereo views remain separate algorithm inputs.

## Pipeline API

```python
from depthai_nodes.node import ImageAlignDepthCorrection
from depthai_nodes.utils.axial_depth import AxialDepthPolicy

correction = pipeline.create(ImageAlignDepthCorrection).build(
    alignedDepth=align.outputAligned,
    sourceDepth=align.passthroughInput,
    policy=AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION,
)
correction.out.link(cloud.inputDepth)
results = correction.result.createOutputQueue()
```

`sourceDepth` is ImageAlign's exact passthrough of the depth frame used for
alignment. A native left image cannot substitute for rectified NeuralDepth
geometry. `alignedDepth.getTransformation()` describes the actual target grid,
so a third RGB input is unnecessary. ImageAlign owns the target-image match.
The benchmark must separately retain native image and replay occurrence IDs.

The node uses bounded blocking inputs and ordered lossless pairing. Source
sequence number and host and device timestamps must match exactly. A dropped
or mismatched pair fails visibly. Keep these streams lossless and preserve
unique occurrence sequence numbers when replaying.

`out` emits a native `dai.ImgFrame` for PointCloud. `result` emits an
`AxialDepthFrameResult` containing that frame and its provenance. The structured
message travels within one Python runtime, including an OAK-4 application's
CPU. Its Python fields do not serialize through XLink or RemoteConnection.
Serialize `result.provenance.to_dict()` into benchmark reports explicitly.

## Computational API

```python
from depthai_nodes.utils.axial_depth import (
    AxialDepthCorrector,
    AxialDepthPolicy,
    geometry_from_frames,
)

corrector = AxialDepthCorrector(
    policy=AxialDepthPolicy.SOURCE_AXIAL_REQUIRES_CORRECTION,
)
result = corrector.correct_frame(aligned_frame, source_depth_frame)
corrected_frame = result.frame
report_fields = result.provenance.to_dict()

geometry = geometry_from_frames(aligned_frame, source_depth_frame)
array_result = corrector.correct(aligned_frame.getFrame(), geometry)
```

`AxialDepthGeometry` accepts `target_size=(width, height)`, `target_intrinsics`,
and `source_from_target_mm`. The frame adapter derives these from the actual
messages and retains their full geometry identity. Array-only callers own the
meaning and calibration of their supplied geometry.

The corrector owns a bounded denominator cache. Defaults are four entries and
64 MiB. Its key includes geometry identity, rather than only image dimensions.
Per-frame geometry changes therefore cannot silently reuse old ray terms.

## Policies and provenance

| Policy                             | Input values                        | Output values                   |
| ---------------------------------- | ----------------------------------- | ------------------------------- |
| `source_axial_requires_correction` | Source depth optical-camera axial Z | Target optical-camera axial Z   |
| `target_axial_passthrough`         | Already target-camera axial Z       | Unchanged target-camera axial Z |

Choose the policy explicitly. There is no SDK-version heuristic. Passthrough
is for a verified producer that already supplies target Z. An SDK upgrade
requires new host and OAK-4 behavioral evidence and an explicit profile change.

Provenance records `correction_enabled`, `correction_method`, the input and
output axial conventions, geometry identity, invalid-value and quantization
rules, rejected count, and frame association. The enabled method is
`source_axial_to_target_axial/1`.

Passing a structured corrected result back into correction rejects double
conversion. Array callers or callers retaining a bare native frame can supply
its `previous_provenance` explicitly. A bare RAW16 frame has no durable axial
convention field. Native cloning drops arbitrary Python attributes. Retain the
matching structured result or sidecar whenever corrected frames cross that
boundary. Changing the configured policy cannot discover an unmarked frame's
history.

## Geometry and validity

For `T_source_target = [R | t]` and the integer target pixel ray
`r = inverse(K_target) [u, v, 1]`, the conversion is
`Z_target = (Z_source - t_z) / (R[2] dot r)`.

The implementation uses float64 arithmetic and nearest-integer millimeter
rounding with ties to even. Source quantization contributes up to
`0.5 / denominator` mm of target error, in addition to output rounding.
Source intrinsics determine upstream alignment sampling. Source pose and
target intrinsics determine the axial conversion.

The frame adapter validates actual source and target sizes, intrinsics, rigid
poses, compatible calibration references, and RAW16 layout. Distorted or
non-perspective target grids fail. Branch RGB undistortion before alignment
when needed. Do not replace the native algorithm RGB view with that branch.

Depth `0` and `65535` remain invalid sentinels. Newly invalid conversions become
zero. These include nonfinite results, nonpositive denominators, nonpositive
rounded depth, and depth outside `1..65534`. Source buffers remain unchanged.
Output cloning preserves the aligned target transformation, timestamps,
sequence, instance, category, source metadata, active-plane offset, row stride,
and padding bytes.

PointCloud consumers must exclude both sentinel depths and nonfinite points.
Native unorganized PointCloud omits zero but can retain saturated depth. Keep
the corresponding valid-depth mask and raster order when filtering. Configure
millimeter units and the RGB optical coordinate frame explicitly. A common
calibration reference other than RGB requires the corresponding cloud-frame
configuration or coordinate conversion before parcel consumers.

Detection and measurement receive an already corrected cloud. Existing box
nodes and the frozen parcel implementations do not perform this correction.

## Verification artifacts

The [pipeline example](../examples/image_align_depth_correction/README.md)
tests actual ImageAlign source-Z behavior and corrected PointCloud coordinates.
The [handoff](axial-depth-correction-handoff.md) records the package artifact,
verification results, and remaining hardware or production acceptance work.
