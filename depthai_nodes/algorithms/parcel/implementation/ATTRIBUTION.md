# Frozen parcel implementation

Luxonis BoxLabRunners `fast_baseline_oak4_visibility_01/v002`, source `d5e35e17d75f91368889dc53d950792eed4096e6`.
Candidate `algorithm-5e4abc3a09b54148950317cdeac60ba2` replaces
`algorithm-2f65a6f6296d48e981f5fe3f77bfde89` in both parcel nodes.

`factory.py`, `geometry.py`, `visibility.py`, and `numpy_ops.py` are the
frozen computation with only imports relocated and repository formatting.
`scene.py` and `components.py` retain the neutral computation boundary; no
Workbench transport, identity registration, journal, or CLI is packaged.
The lazy entry point keeps SciPy optional until computation is constructed.
No custom native kernel, compiler, or binary is used by this candidate.

Settings are byte-identical to the frozen v002 file; `configuration.json`
contains its unchanged parameter object. The source archive digest is not
an OCI image. See manifest.json for original and port hashes separately.
Visualization links alone point to commit-pinned upstream evidence.

The pinned BoxLabRunners source has no LICENSE file. Luxonis attribution is
retained; DepthAI Nodes keeps its Apache-2.0 distribution license.
