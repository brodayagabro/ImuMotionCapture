# IMU / Noitom comparison — iteration 1

## Reuse

| Existing component | Reuse |
| --- | --- |
| `OpenGLMotionCaptureWindow` | ESP32 connection, commands, settings, window lifecycle |
| `MotionCaptureModel` | UDP parsing, sensor mapping, calibration, filtering, orientations |
| `CalibrationWindowMixin` and guided dialogs | A → T → A experiment workflow, existing profile import/export |
| `OpenGLHumanCanvas`, `bone_mesh`, `compute_body_pose` | One view, two independent copies of the same mannequin |
| `RecordingWindowMixin` | Existing optional own BVH export, independent of experiment recording |
| Local `scripts/MocapApi/demo/demo-py/MocapApi/mocap_api/mocap_api.py` | Tested versioned C procedure tables and polling sequence, loaded lazily using ctypes |

## Acquisition and recording

The existing UDP receiver remains responsible for the socket. Add small generic
timestamp/processed-packet hooks to the base viewer, with no Noitom imports.
Use integer `monotonic_ns` at receipt and carry it through the queue unchanged;
convert to seconds only at the existing model/BVH boundary. An experiment packet
claim keeps a recording session alive until its queued packet has been processed.
Stopping recording closes admission; pending claimed packets still reach the disk
worker before metadata is finalized. The GUI never writes session files.

Noitom connect, start, stop and cleanup run in a dedicated worker. Prefer the
DLL bundled with the working Python demo (0.0.73 on Windows), ahead of the old
bin DLL (0.0.17). Match the demo defaults: UDP 7012, binary BVH, transformation
enabled, XYZ. Poll the event count with a null buffer, then fetch initialized
events; a single fixed-buffer poll returned no events on the live stream.
Record the polling delivery mode and flag
SDK frame-index discontinuities as incomplete; an uncached SDK can coalesce
updates before delivery, which the application cannot reconstruct.
Only short avatar/joint getters retain the GIL; lifecycle and polling calls
release it. Releasing it on every joint getter starved the receiver under GUI
and writer load. Convert each recording snapshot to independent JSON-compatible
values once, avoiding repeated full hierarchy copies in the writer.
Timestamp and claim immediately after polling, then copy
the avatar on the SDK receiver thread before the next poll. A Python queue transfers
these independent snapshots to the worker. Keep SDK frame index and timestamp
separately. Both sources use claims to avoid finalizing a session while a received
packet/frame is still being processed; Stop closes admission atomically.
The recorder has independent streams, one T0, no resampling and no index pairing.
JSONL preserves full raw datagrams, SDK joint hierarchy, positions, quaternions,
mapped segment orientations and integer timestamps. Session/event metadata tracks
errors, missing streams and source stops. Render uses only the latest poses.

## Coordinates and retargeting

Select and verify the SDK Default render preset: right handed, Y up, centimetres.
Custom render settings caused Error_NotSupported on live joint rotation reads.
Convert positions to metres and project basis with `0.01 * (-x, z, y)`;
SDK xyzw becomes project wxyz `(w, -x, z, y)`, normalized. Retain both raw SDK
values and converted local transforms in each recorded joint.
BVH joint rotations are local: compose the parent hierarchy to obtain world
orientations. Map Spine2, LeftArm, LeftForeArm, RightArm, RightForeArm using the
SDK joint tags/names, not LeftShoulder/RightShoulder (clavicles).
Use SDK default child offsets, falling back to current parent-local BVH child
translations when default offsets return NotSupported, to derive a direction
correction from the project's A-pose segment vector into the source bone direction. Apply the
world rotation to that correction; this handles the source T-pose without
calibrating away the live motion. All conversions live in the adapter.
Both copies use `compute_body_pose` and the same fixed pelvis and lengths.
Noitom is red wireframe over blue filled own bones, so coincident surfaces remain
legible. Root translation is saved but not rendered.

## Boundaries and validation

The base viewers must import and run without the SDK. Missing/incompatible SDK,
source errors, stale data and recorder errors are visible without terminating
the other receiver. No accuracy scoring, DTW/RMSE or semaphore words.
Hardware-free tests cover basis/quaternion conversion, hierarchy/rest offsets,
asynchronous streams, T0/cutoffs, pending packet draining, repeated sessions,
failure metadata, fake Noitom and loopback UDP. A synthetic mode exercises the
dual canvas. A bounded headless probe and native OpenGL window have also been
tested against live Axis Studio UDP 7012 data, including full 59-joint snapshots,
five mapped segments, recording and stream restart. Physical heading/pose
agreement between suits remains a hardware acceptance step. Replay consumes
the saved orientation mappings without the SDK.
