#!/usr/bin/env python3
"""Process one take with native SkellySync / FreeMoCap / Blender code.

This adapter controls native tasks and verifies their outputs. It does not
implement calibration, tracking, triangulation, filtering or BVH export.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import tomllib


def save(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def synchronize(folder: Path, threshold: float, report):
    import cv2
    import numpy as np
    from skelly_synchronize.core.brightness import compute_brightness_series, find_first_brightness_change
    from skelly_synchronize.core.models import SyncRequest, SyncMethod, VideoBackendKind, VideoInfo
    from skelly_synchronize.core.pipeline.runner import SyncPipeline
    from skelly_synchronize.core.pipeline.stages import BrightnessLagStage

    class CorrectedBrightnessLagStage(BrightnessLagStage):
        """Compatibility fix for skellysync 0231c41, not a new sync detector.

        Upstream emits max(event_times)-event_time but TrimStage *removes*
        lag_seconds from the beginning. Removing event_time-min(event_times)
        aligns the events. A 15-frame fixture otherwise becomes 30 frames apart.
        """
        def run(self, context, progress_callback):
            context = super().run(context, progress_callback)
            largest = max(lag.lag_seconds for lag in context.lags)
            context.lags = [lag.model_copy(update={"lag_seconds": largest-lag.lag_seconds})
                            for lag in context.lags]
            return context

    raw = folder / "raw_videos"
    paths = sorted(raw.glob("*.mp4"))
    if len(paths) < 2:
        raise ValueError("At least two finalized camera videos are required")
    evidence = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if not fps > 0 or count < 10:
            raise ValueError(f"Unreadable or too short recording: {path.name}")
        info = VideoInfo(filepath=path, video_name=path.stem, fps=fps,
                         frame_count=count, duration_seconds=count / fps)
        series = compute_brightness_series(info)
        event = find_first_brightness_change(series, fps, threshold)
        # Upstream falls back to ANY largest change if no threshold is crossed.
        # Reject that fallback: movement/noise must not masquerade as a sync flash.
        delta = np.diff(series, prepend=series[0])
        metric = delta * np.diff(delta, prepend=delta[0])
        if event is None or metric[event.frame_index] < threshold:
            raise ValueError(f"{path.name}: no flash above threshold {threshold:g}. "
                             "Use a brighter common flash; raw files have been preserved.")
        evidence.append(dict(camera=path.stem, frame=event.frame_index,
                             seconds=event.lag_seconds, metric=float(metric[event.frame_index])))
    save(folder / "flash_evidence.json", evidence)
    report("synchronizing", "SkellySync: alignment by the first common flash")
    request = SyncRequest(
        raw_video_folder_path=raw,
        synchronized_video_folder_path=folder / "synchronized_videos",
        method=SyncMethod.BRIGHTNESS, video_handler=VideoBackendKind.FFMPEG,
        brightness_ratio_threshold=threshold, create_debug_artifacts=True)
    pipeline = SyncPipeline.for_request(request)
    pipeline.stages = [CorrectedBrightnessLagStage() if isinstance(stage, BrightnessLagStage) else stage
                       for stage in pipeline.stages]
    result = pipeline.run(request,
        progress_callback=lambda name, fraction: report("synchronizing", f"{name}: {fraction:.0%}"))
    aligned = []
    for video in result.videos_after:
        event = find_first_brightness_change(compute_brightness_series(video), video.fps, threshold)
        if event is None:
            raise ValueError("Flash missing after native synchronization")
        aligned.append(event.frame_index)
    if max(aligned) - min(aligned) > 1:
        raise ValueError(f"Flash alignment check failed after synchronization: {aligned}")
    data = result.model_dump(mode="json")
    data.update(aligned_flash_frames=aligned, compatibility_fix="skellysync-0231c41-brightness-trim-sign")
    save(folder / "sync_result.json", data)
    return result


def native_pipeline(folder: Path, spec: dict, report):
    from skellycam.core.ipc.process_management.worker_registry import WorkerRegistry
    from skellycam.core.ipc.process_management.managed_worker import WorkerMode
    from skellycam.core.recorders.videos.recording_info import RecordingInfo
    from freemocap.core.pipeline.posthoc.posthoc_pipeline_manager import PosthocPipelineManager
    from freemocap.core.pipeline.posthoc.progress_messages import AggregatorNodeProgressMessage
    from freemocap.core.tasks.calibration.calibration_task_config import PosthocCalibrationPipelineConfig
    from freemocap.core.tasks.mocap.mocap_task_config import PosthocMocapPipelineConfig

    if spec["kind"] == "motion":
        from skellycam.core.recorders.videos.parse_video_filename import ParsedVideoFilename
        import cv2
        calibration = tomllib.loads(Path(spec["calibration"]).read_text())
        for path in (folder / "synchronized_videos").glob("*.mp4"):
            name = ParsedVideoFilename.from_path(path).camera_id
            camera = next((value for key, value in calibration.items() if isinstance(value, dict)
                           and value.get("name", key) == name and "matrix" in value), None)
            if camera is None:
                raise ValueError(f"Camera {name} is missing from the selected calibration")
            cap = cv2.VideoCapture(str(path))
            try:
                size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            finally:
                cap.release()
            if list(camera.get("size", [])) != size:
                raise ValueError(f"{name}: video size {size} differs from the calibration {camera.get('size')}")
        from skellytracker.core.detectors.keypoint_detectors.mediapipe import mediapipe_model_manager as models
        report("preparing_models", "Preparing pretrained MediaPipe body, hand and face models")
        models.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        models.get_pose_model_path(models.MediapipePoseModelComplexity.HEAVY)
        models.get_hand_model_path()
        models.get_face_model_path()

    kill = mp.Value("b", False)
    registry = WorkerRegistry(global_kill_flag=kill, worker_mode=WorkerMode.PROCESS)
    registry.start_heartbeat()
    manager = PosthocPipelineManager(global_kill_flag=kill, worker_registry=registry)
    info = RecordingInfo(recording_name=folder.name, recording_directory=str(folder.parent))
    complete = False
    try:
        if spec["kind"] == "calibration":
            manager.create_calibration_pipeline(recording_info=info, calibration_config=
                PosthocCalibrationPipelineConfig(charuco_board=spec["board"], solver_method="anipose"))
        else:
            calibration = Path(spec["calibration"])
            if not calibration.is_file():
                raise ValueError("Select an existing native FreeMoCap calibration TOML")
            manager.create_mocap_pipeline(recording_info=info, mocap_config=PosthocMocapPipelineConfig(
                detector_type="mediapipe", calibration_toml_path=str(calibration),
                export_to_blender=False, auto_open_blend_file=False))
        while manager.pipelines:
            messages = manager.get_progress_updates() + manager.evict_completed()
            for message in messages:
                report(str(message.phase), message.detail, native=asdict(message))
                if str(message.phase) == "failed":
                    raise RuntimeError(message.detail or "Native FreeMoCap pipeline failed")
                if isinstance(message, AggregatorNodeProgressMessage) and str(message.phase) == "complete":
                    complete = True
            time.sleep(0.1)
        if not complete:
            raise RuntimeError("Native pipeline exited without a successful aggregator completion")
    finally:
        manager.shutdown()
        registry.shutdown_all()


def processing_timeline(folder: Path):
    """Describe the *resampled output timeline*, never invent capture timestamps.

    Native filters and Blender read a timestamps CSV rather than MP4 metadata.
    Original SkellyCam timestamps remain untouched under camera_files/.
    """
    import cv2
    metadata = []
    for path in sorted((folder / "synchronized_videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        try:
            metadata.append((cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        finally:
            cap.release()
    if not metadata or any(fps <= 0 or count < 12 for fps, count in metadata):
        raise ValueError("Need at least 12 synchronized frames with valid FPS")
    fps, count = metadata[0]
    if any(abs(rate-fps) > 1e-6 or frames != count for rate, frames in metadata):
        raise ValueError("Synchronized videos have inconsistent frame rates/counts")
    target = folder / "synchronized_videos" / "timestamps"
    target.mkdir(exist_ok=True)
    with (target / "normalized_timeline_timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "normalized_time_seconds", "from_previous.frame_duration.ms", "from_previous.framerate.hz"])
        writer.writerows((i, i/fps, 1000/fps, fps) for i in range(count))
    save(target / "provenance.json", dict(source="SkellySync output constant-frame-rate timeline",
         hardware_capture_timestamps=False, fps=fps, frame_count=count,
         original_timestamps="../../camera_files"))


def run(spec: dict, report):
    folder = Path(spec["folder"]).resolve()
    if not (folder / "sync_result.json").is_file():
        synchronize(folder, spec.get("brightness_threshold", 1000.0), report)
    if spec["kind"] == "sync":
        return dict(sync=str(folder / "sync_result.json"))
    processing_timeline(folder)
    native_pipeline(folder, spec, report)
    if spec["kind"] == "calibration":
        paths = list(folder.glob("*.toml"))
        if len(paths) != 1 or not paths[0].stat().st_size:
            raise RuntimeError("Native calibration did not produce a unique TOML")
        return dict(calibration=str(paths[0]))
    from freemocap.core.blender import export_to_blender as native_export
    report("exporting_blender", "Blender: native FreeMoCap armature and BVH export")
    def launch_blender(command_list):
        # Native exporter prepends the host's site-packages inside Blender.
        # Load Blender's own NumPy first: its Python ABI may differ from the host.
        # A distro Blender must also find system Python, not uv/conda's executable.
        env = dict(os.environ)
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
        command = [command_list[0], "--factory-startup", "--python-expr", "import numpy", *command_list[1:]]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    original = native_export.run_subprocess
    native_export.run_subprocess = launch_blender
    try:
        native_export.export_to_blender(recording_folder_path=folder, detector="mediapipe",
                          blender_exe_path=spec["blender"], open_file_on_completion=False,
                          blender_export_config={"formats": ["bvh"]})
    finally:
        native_export.run_subprocess = original
    bvh = list((folder / "3d_models").glob("*.bvh"))
    if len(bvh) != 1 or not bvh[0].stat().st_size:
        raise RuntimeError("Blender did not produce a unique nonempty BVH; inspect worker.log")
    # Native task treats Blender as optional; this adapter requires actual BVH.
    text = bvh[0].read_text()
    if "MOTION" not in text or "Frame Time:" not in text:
        raise RuntimeError("Invalid native BVH output")
    return dict(bvh=str(bvh[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    folder = Path(spec["folder"])
    os.environ["FREEMOCAP_BASE_FOLDER"] = str(folder / "native_runtime")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)

    def report(phase, detail, **extra):
        save(folder / "progress.json", dict(phase=phase, detail=detail, **extra))

    try:
        result = run(spec, report)
        save(folder / "result.json", result)
        report("complete", "Native processing complete")
    except Exception as exc:
        report("failed", f"{type(exc).__name__}: {exc}")
        raise
