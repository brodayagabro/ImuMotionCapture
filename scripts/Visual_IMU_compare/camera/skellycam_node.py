#!/usr/bin/env python3
"""LAN control and file transfer around the unmodified SkellyCam recorder.

One process / USB camera. Run with .venv-skellycam/bin/python; do not run the
RTSP streamer against the same device. Camera exposure is NOT network-triggered.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def identifier(value: str) -> str:
    if not NAME.fullmatch(value):
        raise ValueError("Use 1–64 letters, digits, underscores or hyphens")
    return value


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def device_index(device: str) -> int:
    if device.isdecimal():
        return int(device)
    path = Path(device).resolve(strict=True)
    match = re.fullmatch(r"video(\d+)", path.name)
    if path.parent != Path("/dev") or not match:
        raise ValueError(f"Expected a V4L2 camera device, got {path}")
    return int(match[1])


def file_manifest(folder: Path) -> list[dict]:
    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name.startswith(".node-"):
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append(dict(path=path.relative_to(folder).as_posix(), size=path.stat().st_size, sha256=digest))
    return files


class NativeRecorder:
    def __init__(self, args):
        from skellycam.core.ipc.process_management.worker_registry import WorkerRegistry
        from skellycam.core.ipc.process_management.managed_worker import WorkerMode
        from skellycam.core.camera_group.camera_group_manager import CameraGroupManager
        self.args = args
        self.kill = mp.Value("b", False)
        self.registry = WorkerRegistry(global_kill_flag=self.kill, worker_mode=WorkerMode.PROCESS)
        self.registry.start_heartbeat()
        self.manager = CameraGroupManager(global_kill_flag=self.kill, worker_registry=self.registry)
        self.ready = False

    async def prepare(self):
        from skellycam.core.camera.config.camera_config import CameraConfig
        from skellycam.core.recorders.videos.fourcc_codec_helpers import PYAV_H264_FOURCC
        if not self.ready:
            width, height = map(int, self.args.resolution.split("x"))
            config = CameraConfig(camera_id=self.args.name, camera_name=self.args.name,
                                  camera_index=device_index(self.args.device),
                                  resolution=dict(width=width, height=height), framerate=self.args.fps,
                                  exposure_mode=self.args.exposure_mode, exposure=self.args.exposure,
                                  capture_fourcc=self.args.capture_fourcc, writer_fourcc=PYAV_H264_FOURCC)
            await self.manager.create_or_update_camera_group({self.args.name: config})
            self.ready = True

    async def start(self, take_id: str):
        from skellycam.core.recorders.videos.recording_info import RecordingInfo
        await self.manager.start_recording_all_groups(RecordingInfo(
            recording_name=take_id, recording_directory=str(self.args.records), mic_device_index=-1))

    async def stop(self):
        await self.manager.stop_recording_all_groups()

    async def close(self):
        try:
            await self.manager.close_all_camera_groups()
        finally:
            self.registry.shutdown_all()


def create_app(args, recorder_factory=NativeRecorder):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    args.records = Path(args.records).expanduser().resolve()
    args.records.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    active = None
    recorder = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal recorder
        recorder = recorder_factory(args)
        try:
            yield
        finally:
            if active is not None:
                try:
                    await recorder.stop()
                    atomic_json(args.records / active / ".node-state.json", dict(state="closed"))
                except Exception:
                    pass  # Interrupted records remain local, never advertised as finalized.
            await recorder.close()

    app = FastAPI(title="SkellyCam recording node", lifespan=lifespan)

    def take_path(take_id):
        try:
            return args.records / identifier(take_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/health")
    async def health():
        return dict(camera=args.name, device=args.device, resolution=args.resolution,
                    fps=args.fps, active=active, ready=bool(recorder and recorder.ready),
                    engine="skellycam", protocol=1)

    @app.post("/prepare")
    async def prepare():
        async with lock:
            try:
                await recorder.prepare()
            except Exception as exc:
                raise HTTPException(503, str(exc)) from exc
            return await health()

    @app.post("/recordings/{take_id}/start")
    async def start(take_id: str):
        nonlocal active
        folder = take_path(take_id)
        async with lock:
            if active == take_id:
                return dict(state="recording", camera=args.name)
            if active or folder.exists():
                raise HTTPException(409, "Camera busy or recording already exists")
            await recorder.prepare()
            folder.mkdir()
            atomic_json(folder / ".node-state.json", dict(state="starting"))
            active = take_id  # Keep recoverable identity even if native start fails.
            try:
                await recorder.start(take_id)
            except Exception as exc:
                raise HTTPException(503, str(exc)) from exc
            atomic_json(folder / ".node-state.json", dict(state="recording", started_unix=time.time()))
            return dict(state="recording", camera=args.name)

    @app.post("/recordings/{take_id}/stop")
    async def stop(take_id: str):
        nonlocal active
        folder = take_path(take_id)
        async with lock:
            if active == take_id:
                await recorder.stop()
                active = None
                atomic_json(folder / ".node-state.json", dict(state="closed"))
            state = folder / ".node-state.json"
            if not state.is_file() or json.loads(state.read_text()).get("state") != "closed":
                raise HTTPException(409, "Recording not finalized; inspect the camera node log")
            return dict(state="closed")

    @app.get("/recordings/{take_id}/manifest")
    async def manifest(take_id: str):
        folder = take_path(take_id)
        async with lock:
            state = folder / ".node-state.json"
            if active == take_id or not state.is_file() or json.loads(state.read_text()).get("state") != "closed":
                raise HTTPException(409, "Recording is not finalized")
            return dict(camera=args.name, files=await asyncio.to_thread(file_manifest, folder))

    @app.get("/recordings/{take_id}/files/{relative:path}")
    async def download(take_id: str, relative: str):
        folder = take_path(take_id)
        state = folder / ".node-state.json"
        if not state.is_file() or json.loads(state.read_text()).get("state") != "closed":
            raise HTTPException(409, "Recording is not finalized")
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or path.name.startswith(".node-"):
            raise HTTPException(404)
        return FileResponse(path)

    return app


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--name", type=identifier, required=True, help="Unique camera identity, e.g. cam1")
    result.add_argument("--device", default="/dev/video0", help="V4L2 index or persistent /dev/v4l/by-id path")
    result.add_argument("--resolution", default="1280x720")
    result.add_argument("--fps", type=float, default=30)
    result.add_argument("--capture-fourcc", default="MJPG", choices=["MJPG", "YUYV"])
    result.add_argument("--exposure-mode", default="AUTO", choices=["AUTO", "MANUAL"])
    result.add_argument("--exposure", type=int, default=-7)
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=8082)
    result.add_argument("--records", type=Path, default=HERE / "runtime" / "records")
    return result


if __name__ == "__main__":
    import uvicorn
    args = parser().parse_args()
    if not re.fullmatch(r"[1-9]\d{1,4}x[1-9]\d{1,4}", args.resolution) or not 0 < args.fps <= 240:
        raise SystemExit("Invalid resolution or FPS")
    os.environ.setdefault("SKELLYCAM_BASE_FOLDER", str(HERE / "runtime" / "skellycam"))
    mp.set_start_method("spawn", force=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port)
