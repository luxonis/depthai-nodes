# Fast baseline implementation checklist

The accepted handoff fixes the algorithm and node boundary. This checklist tracks implementation and verification, without reopening algorithm selection.

- [x] Read applicable instructions, contribution requirements, accepted design, and promotion runbook.
- [x] Ground the affected subsystem. Existing box nodes retain their tracked masks and temporal measurement behavior. Parcel nodes use separate scene and parcel messages.
- [x] Confirm the architecture. A completed scene call produces independent parcel arrays and a plane snapshot. The dimensioner consumes that batch.
- [x] Blocking first steps. Match all 16 source hashes to `0a2f571843f81ac6cc2f8cebe3475540e314f83d` and scoped settings to `4b7634cc5806b7fb2e48bc7a830a3434b85cc0cb` before porting.
- [x] Independent workstreams. Separate immutable computation and packaging, node/message conversion, and reference comparison tooling.
- [x] Shared mutable state. Each detector owns its components and completes a scene before processing another. Workers own disjoint files. Final hash generation follows all source edits.
- [x] Smallest safe decomposition. Three implementation workers share the settled contract. The coordinator owns integration, documentation, installed-wheel checks, and final review.
- [x] Implement the immutable computation and manifest.
- [x] Implement the node and message boundary, examples, and focused tests.
- [x] Compare frozen computation and port on synthetic and authorized validation scenes.
- [x] Run unit, lint, documentation, and installed-wheel pipeline checks. Record existing full-suite failures separately.
- [x] Review the final changes and record limitations. Two independent reviews found no outstanding code defects after fixes.
- [x] Prepare one cohesive verified change for a local commit. Splitting the binding from the node files would create an invalid intermediate manifest. Publishing and merging are outside this handoff.

Model roles inherit the current session. The installed skill's default model families are unavailable in this environment. Independent workers do not establish model diversity.
