# 0001 Support plane acceptance and selection

Recorded 2026-11-13.

## Scope

Design `SupportPlaneCalibration` for fixed cameras and fixed surfaces such as a conveyor. Commission the installation under human review. Temporary occlusion can reduce measured support for the plane. Report that evidence when asked to verify, and let the application decide what it means for operation.

Camera movement detection belongs to a separate component. When that component reports a move, the application invalidates the active approval, pauses plane-dependent processing, and starts a new human-reviewed calibration. Continuous monitoring and automatic recalibration are outside this node. Moving platforms such as forklifts are outside the initial design.

## Context

The support plane workflow in PR #340 discovers planes using RANSAC inliers on a filtered cloud. `Verify` checks a supplied equation using coverage measured on a grid sample of the original cloud. The two paths can disagree on the same input. `AutoSupportPlane` can therefore accept a newly fitted plane that it would reject when restored as `initialPlane`.

## Decision

- **Fit hypotheses.** Propose plane equations from the cloud. Fitting can use its own constraints and inlier statistics to decide which equations are worth evaluating.
- **Verify geometry.** Evaluate a supplied equation against the cloud and gravity. Report whether it is sufficiently level, below the camera, and supported by enough of the cloud. Expose measurements that other steps can use, such as the vertical distance from the camera to the plane.
- **Check application requirements.** Evaluate configured requirements such as a vertical distance range of 70–90 cm. Report which requirement failed so callers can distinguish an unsupported equation from a supported plane outside the application's bounds.
- **Select or approve.** Downstream code decides which equation to use, programmatically or through human review. It can use the assessment, rank planes by a preference such as proximity to 80 cm, or apply its own policy.

## Pipeline communication

The support plane workflow uses typed messages through DepthAI node inputs and outputs. The node API must not accept user callbacks for selection or approval. Applications and downstream nodes consume assessment messages and can send decisions through an input when the workflow needs a reply. This keeps application logic outside the calibration node's execution loop and allows human review to proceed asynchronously.

## Public interface

The names below define the proposed interface. `SupportPlaneCalibration.build(pointCloud, imu, pointCloudSize, config=None)` links the organized millimeter cloud and IMU outputs. `pointCloudSize` is `(width, height)`. IMU input must provide an acceleration direction in the point cloud's coordinate frame.

| Port                 | Direction | Contract                                                                                                                |
| -------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------- |
| `pointCloud`         | Input     | Organized `dai.PointCloudData` in millimeters.                                                                          |
| `imu`                | Input     | `dai.IMUData` with an accelerometer report.                                                                             |
| `inputControl`       | Input     | `SupportPlaneCalibrationControl.start()` and `.stop()` construct typed `START` and `STOP` messages.                     |
| `inputVerification`  | Input     | `SupportPlaneVerificationRequest(plane, request_id=None)` requests one assessment.                                      |
| `candidatesOutput`   | Output    | `SupportPlaneCandidateBatch.assessments` carries zero or more `PlaneAssessment` values from one fitting attempt.        |
| `verificationOutput` | Output    | `SupportPlaneVerificationResult.assessment` carries one `PlaneAssessment`. `request_id` echoes the optional request ID. |

`SupportPlaneCalibrationConfig` exposes these settings. The first controls search scheduling. The remaining settings apply to the shared evaluator in both paths.

| Setting                      | Default | Meaning                                                                              |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `attempt_interval_sec`       | `0.5`   | Minimum time between fit starts. Must be positive and finite.                        |
| `max_gravity_angle_deg`      | `6.5`   | Maximum tilt from horizontal.                                                        |
| `check_distance_mm`          | `30.0`  | Maximum perpendicular distance for a sampled point to count as on-plane.             |
| `min_sampled_point_count`    | `500`   | Minimum number of valid grid samples needed to judge coverage.                       |
| `min_on_plane_ratio`         | `0.10`  | Minimum fraction of valid grid samples on the plane.                                 |
| `vertical_distance_range_mm` | `None`  | Optional inclusive `(min, max)` installation requirement for distance along gravity. |

### Search lifecycle

1. Start idle and keep consuming both sensor inputs. Retain the latest cloud and latest usable gravity reading instead of buffering a backlog. Use one cloud and one gravity reading per attempt. Preserve the existing host-arrival freshness limit of 0.5 seconds for gravity independently of the fitting interval.
1. On `START` while idle, begin searching. The first fit is eligible immediately when inputs are usable. Subsequent fit starts are separated by at least `attempt_interval_sec`. Within a search, fit each cloud at most once. Wait for a new cloud and usable gravity when needed. `START` while searching leaves the current search in progress.
1. For every completed fit, publish exactly one batch. An empty batch means the fit produced no distinct equations to assess. If equations were assessed and all failed, publish them with their reasons and continue. Waiting for input does not produce an attempt batch.
1. Publish the complete batch before returning to idle if any assessment has `accepted=True`. With no passing assessment, search indefinitely until a `STOP` command is processed. `STOP` prevents new fits and returns the search to idle. A fit already running can finish and publish. `STOP` while idle has no effect.
1. A fresh `START` can follow success or cancellation directly. Fitting uses each cloud independently, so the interface has no `RESET` command. If a reviewer declines every plane in a completed search, the application can send another `START`.

Verification is independent of search state and fit timing. Assess one supplied equation per request using the latest usable cloud and gravity. A request can wait for usable inputs. Answer pending requests in arrival order and echo each request ID. Verification never starts, stops, or completes a search and never activates a plane.

The dedicated state output and its message schema are a separate design decision. Candidate batches are evidence about completed fits. In particular, consumers should not use the absence of a batch to distinguish an idle node from one waiting for input.

## Shared plane assessment

Search reports an assessment for each distinct fitted equation it evaluates, including equations that fail. Explicit verification reports one assessment for the supplied equation. Both paths produce the same measurements, verdict, and reasons for the same equation, point cloud, gravity direction, and configuration. Saved planes go through the same checks. After human review, downstream code can request a fresh assessment before using the approved plane.

For both paths, take rows `0, 4, 8, …` and columns `0, 4, 8, …` from the **original organized point cloud**, then keep the sampled points with `z > 0`. Voxel downsampling and denoising are part of equation fitting. The assessment counts and ratio use the grid sample.

| Field                       | Meaning                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `plane`                     | Plane equation evaluated.                                                                                                             |
| `sampled_point_count`       | Number of positive-depth points on the sampling grid.                                                                                 |
| `on_plane_point_count`      | Number of those points whose perpendicular distance to the plane is at or below the configured tolerance.                             |
| `on_plane_ratio`            | `on_plane_point_count / sampled_point_count`. Unavailable when the sample count is zero.                                              |
| `tilt_deg`                  | Tilt from horizontal relative to the gravity direction used.                                                                          |
| `perpendicular_distance_mm` | Shortest distance from the camera origin to the plane.                                                                                |
| `vertical_distance_mm`      | Distance from the camera origin along gravity to the plane. Unavailable when there is no well-defined intersection in that direction. |
| `accepted`                  | Whether the assessment passes all configured checks.                                                                                  |
| `reasons`                   | Structured failure reasons, with measured values and configured limits where applicable. Empty on acceptance.                         |

Report both counts even if too few points were sampled to judge coverage. Distinguish insufficient evidence from a failed check. For example, use `insufficient_sample` when the sample is too small and `coverage_below_minimum` when a sufficient sample has too little support. Report independently evaluable failures together.

## Snapshot provenance

Publish assessments in a typed DepthAI message that preserves the source point cloud's complete `Buffer` metadata. Copy the sequence number, host-synchronized timestamp, device monotonic timestamp, and optional device system timestamp, which may be PTP-synchronized. Use `setBufferMetadataFrom(cloud)` for the copy. Attach the metadata once per evaluated cloud. A search message can carry multiple assessments from that cloud, and an explicit verification message carries one. When multiple cloud streams are involved, include a source stream identifier so consumers can resolve the snapshot.

Record the gravity direction actually used, expressed in the point cloud's coordinate frame, and the timestamp of the IMU sample used to derive it when available. Preserve the IMU timestamp's clock domain separately from the cloud metadata. The cloud and IMU sample can come from different instants.

## Active plane persistence

Provide a reusable, application-owned `SupportPlaneStore(path)` helper. `save(record)` writes the active, explicitly approved plane as versioned JSON using atomic replacement. `load(device_id=..., frame_id=...)` returns the compatible approved record or `None` when no record exists. It reports unreadable files with an I/O error and invalid, unsupported, or incompatible records with a validation error. `invalidate()` clears the stored approval so a restart cannot reactivate it. The calibration node does not read or write application state.

An `ActiveSupportPlane` record contains `plane`, `device_id`, `frame_id`, and an `approved_at` UTC timestamp. The persisted envelope includes `schema_version`. Frame identity must distinguish coordinate frames or installation configurations that make the same numerical equation unsafe to reuse. The application identifies the installation when resolving storage and owns the policy for resuming from a compatible record, requesting verification, or asking for renewed approval. There is no factory support plane. On first startup without a saved approval, plane-dependent processing waits for commissioning.

`resolve_support_plane_path(app_id, installation_id, path=None)` uses an explicit path first. For a standalone OAK app with `OAKAGENT_STORAGE_APP`, default to `$OAKAGENT_STORAGE_APP/support-plane.json`. Use an explicit installation-specific path if one app manages multiple installations. For a host app in peripheral mode, default to the OS per-user state directory, scoped by application and installation. On Linux that is `$XDG_STATE_HOME/<app_id>/<installation_id>/support-plane.json`, with `~/.local/state` as the fallback base. A host service can configure a durable path appropriate to its deployment. [OAK app persistent storage](https://docs.luxonis.com/software-v3/oak-apps/oakctl#persistent-storage)

The model zoo cache is for replaceable downloads. The support plane record holds application state that persists across updates and represents human approval. [DepthAI model cache settings](https://docs.luxonis.com/software-v3/depthai/depthai-components/device)

## Implementation checks

- Exercise `START`, `STOP`, success, and restart without `RESET`. Use a controlled clock and input stream to check the default interval, no duplicate cloud within a search, and no attempt batch while waiting for inputs.
- Check empty, all-rejected, and mixed batches. Confirm that the full mixed batch is published before the search returns to idle.
- Evaluate the same equation and sensor snapshot through both paths. Assert identical assessments, shared sampling, structured reasons, complete cloud metadata, and separate IMU provenance.
- Verify while idle and while searching. Confirm one reply per request, request ID correlation, and no changes to search state due to verification.
- Check first run, compatible restore, incompatible and invalidated records, and explicit path precedence. Keep movement detection and application activation outside the node.
