# Parcel algorithm findings for BoxLabRunners

These findings concern the frozen fast baseline, `fast_baseline/v001`, candidate `algorithm-2f65a6f6296d48e981f5fe3f77bfde89`.
Its evaluated BoxLabRunners source is `0a2f571843f81ac6cc2f8cebe3475540e314f83d`; committed evidence is at `4b7634cc5806b7fb2e48bc7a830a3434b85cc0cb`.
The packaging audit of depthai-nodes PR [#348](https://github.com/luxonis/depthai-nodes/pull/348), through `10f0b79d5473aada6face5d60a2e192c144d56ce`, found no algorithm correction relative to that evaluated source.
Computation ASTs agree after import and docstring adaptations. C++ source and scoped-settings bytes match exactly.

BoxLabRunners owns the corrections, review, configuration tuning, and evaluation.
The stable-directory migration preserves the current computation and its known limitations.
Historical scores stay attached to the original candidate. Corrections require a newly reviewed and evaluated candidate before another promotion.

## Confirmed native projection discrepancy

[CodeRabbit finding 4667613133](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613133) reproduces in both the original native implementation and the port.
Use one float32 XYZ point `[[784, 0, 784]]`, an identity camera pose, a perspective camera with empty distortion coefficients, image size 2000 by 1500, and intrinsics:

```python
((600, 0, 120), (0, 600, 0), (0, 0, 1))
```

With six-pixel depth cells, the fallback projects to `u=720` and writes row zero, column 120.
The native reciprocal multiplication produces a value just below 720 and writes flattened cell 119.
The resulting depth-grid placement can change visibility decisions. This is an inherited algorithm correctness issue, not an import relocation error.
Review any direct-division correction in BoxLabRunners, test cell boundaries, and evaluate the changed combined candidate.
The earlier native/fallback equality claim applies only to the compared scenes, not every supported input.

## Performance findings on OAK-4

The live native-enabled detector took a median 469 ms across 397 measured calls; the dimensioner took 5.92 ms across 398 calls.
A separate depth capture arrived at 10.00 FPS. Detector computation limits throughput under this workload.
The captured scene contained 318,637 XYZ points and 206,668 foreground points.

- The 12 mm foreground voxel bounding grid spans 1394 by 543 by 1042 cells, or 788,733,564 cells. Native detection rejects grids above 8,000,000 cells and falls back to NumPy grouping. A separate diagnostic measured approximately 149 ms in grouping. Raising the dense cap would require approximately 6.31 GB for two int32 arrays alone.
- Clustering has 30,851 centroids. Its conservative neighbor estimate is 10,581,893, selecting the slower batched path. The actual directed neighbor count, including self-neighbors, is 500,389. A separate diagnostic measured approximately 99 ms in clustering.

Temporary prototypes added sparse native voxel processing and selected the existing pair-query clustering path using actual neighbor counts.
Five alternating samples per configuration on the same saved scene measured:

| Configuration                                               | Median checked computation |
| ----------------------------------------------------------- | -------------------------: |
| Current native implementation                               |                  349.72 ms |
| Sparse native voxel processing                              |                  219.85 ms |
| Sparse processing plus actual-neighbor clustering selection |                  178.19 ms |

The complete checked results matched exactly across these 15 runs. Capture and UI were stopped during replay, and each sample used a fresh process, one warm-up, and fresh measured components.
These are one-scene diagnostic results, not Workbench evaluation statistics or measured live FPS for the prototypes.
No prototype is included in the packaged algorithm. The final 178 ms result still falls short of 8 to 10 FPS before application overhead.
The historical approximately 100 ms reference ran in a Linux arm64 container on an Apple M1 Pro and remains attributed to its original artifact.

## Additional algorithm review findings

| Review finding                                                                                                       | Classification, evidence, and effect                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [4667613126](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613126), fallback grid initialization | Frozen factory and visibility scheduling. The unset-grid check has no synchronization, allowing parallel candidates to repeat projection and grid construction. Review an eager or synchronized initialization upstream and evaluate it.                                                                                                                                                               |
| [4667613138](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613138), native allocation failure    | Native runtime failure semantics. The `centered_hull_candidates` vector allocation can throw across the C ABI into ctypes and terminate the host. Source inspection confirms the missing catch; memory exhaustion was not induced. An upstream correction needs a distinct error status and matching wrapper handling.                                                                                 |
| [4667613147](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613147), label-grid copies            | Frozen visibility computation. Boolean indexing copies other labels' layers for each proposal, causing quadratic work. This is a source-level finding; the review's 100-proposal scenario was not benchmarked. Review nearest and second-nearest depth caching upstream with equal-depth and excluded-label tests.                                                                                     |
| [4667613143](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613143), stale native exports         | Ownership crosses installation and frozen loader behavior. A local binary can load yet lack an export, raising `AttributeError` outside the loader's `OSError` handler. The loader is unchanged from upstream. This port excludes native binaries and requires fresh local builds after updates. Robust ABI fallback changes belong in BoxLabRunners because they alter the frozen dependency closure. |

## Integration findings addressed here

The stable-directory migration addresses the following packaging and message-boundary findings without correcting the frozen algorithm:

- [4667613116](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613116) clarifies that scoped settings retain the committed release-scoped artifact's bytes, after the earlier upstream lineage edit.
- [4667613154](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613154) is covered by the existing LF checkout rules in `10f0b79`.
- [4667613163](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613163) gives published parcel arrays immutable backing storage. It adds one message-boundary snapshot and preserves float32 values and point order. Measurement does not copy the arrays again.
- [4667613169](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613169) makes the documented config producer queue blocking and explains backpressure.
- [4667613177](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613177) points native-build instructions at the stable module path.
- [4667613185](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613185) observes worker waiting at the callback boundary instead of relying on exception-handler scheduling.
- [4667613193](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4667613193) checks external reference files against repository-owned provenance before importing them.

The migration also packages and hashes the existing visualization, with separate documentation-source provenance and repaired links.
Relocation and packaging checks do not establish that the production port is the original evaluated artifact.
The [promotion record](fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.md) states the actual source-comparison coverage.
