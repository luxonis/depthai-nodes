# Fast baseline source attribution

The computation was authored by Luxonis in [BoxLabRunners](https://github.com/luxonis/BoxLabRunners)
and ported from commit `0a2f571843f81ac6cc2f8cebe3475540e314f83d`.
The evaluated candidate is `algorithm-2f65a6f6296d48e981f5fe3f77bfde89`.
Evidence and scoped settings come from commit `4b7634cc5806b7fb2e48bc7a830a3434b85cc0cb`.

`geometry.py`, `visibility.py`, `native.py`, `kernels.cpp`, and `build_native.py`
retain the computation. Only Runner imports, formatting, and import placement change.
`factory.py` is the original algorithm factory. `scene.py` retains the neutral
Runner dataclasses and the exact projector. `components.py` retains the two
callback fields without optional plane estimation or Workbench transport.
The lightweight package and binding defer the optional SciPy imports until use.

The source repository did not contain a license file at the pinned revision.
This attribution does not infer a separate upstream license grant. The published
DepthAI Nodes package retains its Apache-2.0 license in the distribution metadata.
No third-party computation source is vendored. NumPy, SciPy, and OpenCV retain
their own distribution licenses.

`scoped-settings.json` and `evaluation-evidence.json` retain the original bytes.
The scoped copy removes an inaccessible historical parent registration declaration.
It preserves the configuration, policy, objective, and modalities. Historical
metrics remain attached to the artifact that produced them.

No native binary is bundled in the universal wheel. Missing or unsupported native
kernels use the original NumPy/OpenCV fallback. Build an optional local binary with
`python -m depthai_nodes.algorithms.parcel.fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.build_native`.
The compiler must support C++17 and `-O3 -ffp-contract=off -fPIC -shared`.
The output is specific to that operating system and architecture. Do not copy the
retained Linux aarch64 reference binary to other platforms.
