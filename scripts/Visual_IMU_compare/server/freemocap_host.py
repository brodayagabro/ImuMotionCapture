#!/usr/bin/env python3
"""RTSP recording → native offline FreeMoCap → BVH + web player."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time
from urllib.parse import urlsplit
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from rtsp_recording import RTSPRecorder, finalize_recordings, probe
from rtsp_preview import Preview

HERE = Path(__file__).resolve().parent
BUSY = {"preparing", "recording", "collecting", "processing"}


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temporary.replace(path)


class StartRequest(BaseModel):
    kind: str = "motion"
    output_directory: str = ""
    filename: str = "motion.bvh"
    calibration: str = ""
    sync_method: Literal["receive", "flash"] = "receive"
    camera_offsets_ms: dict[str, float] = Field(default_factory=dict)
    max_skew_ms: float = Field(default=50, gt=0, le=5000, allow_inf_nan=False)
    max_age_ms: float = Field(default=500, gt=0, le=5000, allow_inf_nan=False)
    brightness_threshold: float = Field(default=1000, gt=0, allow_inf_nan=False)
    squares_x: int = Field(default=5, ge=2, le=30)
    squares_y: int = Field(default=3, ge=2, le=30)
    square_mm: float = Field(default=54, gt=0, le=1000, allow_inf_nan=False)
    marker_ratio: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    dictionary: str = "DICT_4X4_250"


class SaveRequest(BaseModel):
    output_directory: str
    filename: str


def publish_bvh(source: Path, directory: str, filename: str) -> Path:
    if not re.fullmatch(r"[^/\\\x00]{1,120}\.bvh", filename, flags=re.IGNORECASE):
        raise ValueError("Use a plain filename ending in .bvh")
    folder = Path(directory).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / filename
    # Link a fully written temporary file: atomic, and never overwrites a target.
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".bvh-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def camera_endpoints(values):
    endpoints = {}
    for value in values:
        name, separator, url = value.partition("=")
        parsed = urlsplit(url)
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Use --camera cam1=rtsp://SERVER:8554/cam1")
        if (parsed.scheme != "rtsp" or not parsed.hostname or parsed.fragment
                or parsed.username or any(ord(c) < 32 for c in url)):
            raise ValueError("Use an RTSP URL without credentials")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("Invalid camera port")
        if name in endpoints:
            raise ValueError(f"Duplicate camera identity: {name}")
        endpoints[name] = url
    if len(endpoints) < 2:
        raise ValueError("Configure at least two distinct camera nodes")
    if len(set(endpoints.values())) != len(endpoints):
        raise ValueError("Each camera must have a different endpoint")
    return endpoints


class Host:
    def __init__(self, args):
        self.args = args
        self.root = args.records.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cameras = camera_endpoints(args.camera)
        self.transport = "rtsp"
        self.capture = None
        self.capture_watch = None
        self.health_cache = (0, [])
        self.health_lock = asyncio.Lock()
        self.ffmpeg = getattr(args, "ffmpeg", "ffmpeg")
        self.ffprobe = getattr(args, "ffprobe", "ffprobe")
        self.capture_timeout = getattr(args, "capture_timeout", 10)
        self.previews = {name: Preview(url, self.ffmpeg, self.capture_timeout) for name, url in self.cameras.items()}
        if self.transport == "rtsp":
            for tool in (self.ffmpeg, self.ffprobe):
                if not shutil.which(tool):
                    raise ValueError(f"Missing {tool}; install server FFmpeg / ffprobe")
        self.takes = {}
        self.lock = asyncio.Lock()
        self.task = None
        self.process = None
        for path in self.root.glob("*/take.json"):
            take = json.loads(path.read_text())
            if take["state"] in BUSY:
                take.update(state="interrupted", error="Host restarted. Stop/collect again to recover the take.")
                save(path, take)
            self.takes[take["id"]] = take

    def update(self, take, **values):
        take.update(values)
        save(self.root / take["id"] / "take.json", take)

    def get(self, take_id):
        if take_id not in self.takes:
            raise HTTPException(404, "Unknown recording")
        return self.takes[take_id]

    async def health(self):
        async with self.health_lock:
            active = next((t for t in self.takes.values() if t["state"] == "recording"), None)
            if active and self.capture:
                return [dict(active["camera_settings"][name], active=active["id"],
                             received_seconds=stream.media_time, error=stream.failure())
                        for name, stream in self.capture.streams.items()]
            if time.monotonic() - self.health_cache[0] < 10:
                return self.health_cache[1]
            async def stream_health(name, url):
                try:
                    info = await probe(url, self.ffprobe, self.capture_timeout)
                    return dict(camera=name, engine="rtsp", ready=True, active=None, **info)
                except Exception as exc:
                    return dict(camera=name, engine="rtsp", error=str(exc) or type(exc).__name__)
            status = await asyncio.gather(*(stream_health(n, u) for n, u in self.cameras.items()))
            self.health_cache = (time.monotonic(), status)
            return status

    async def start(self, request):
        async with self.lock:
            if any(t["state"] in BUSY for t in self.takes.values()):
                raise HTTPException(409, "A recording or processing task is already active")
            if request.kind not in ("sync", "calibration", "motion"):
                raise HTTPException(400, "Unknown recording kind")
            if (request.max_skew_ms > request.max_age_ms or set(request.camera_offsets_ms) - set(self.cameras)
                    or any(not math.isfinite(v) or abs(v) > 5000 for v in request.camera_offsets_ms.values())):
                raise HTTPException(400, "Invalid camera timing offsets or max_skew_ms exceeds max_age_ms")
            if not re.fullmatch(r"[^/\\\x00]{1,120}\.bvh", request.filename, flags=re.IGNORECASE):
                raise HTTPException(400, "Use a plain filename ending in .bvh")
            if request.kind == "motion":
                calibration = Path(request.calibration).expanduser().resolve()
                if not calibration.is_file() or calibration.suffix != ".toml":
                    raise HTTPException(400, "Select a native FreeMoCap calibration TOML")
                if not shutil.which(self.args.blender):
                    raise HTTPException(400, "Blender executable not found; set --blender")
                request.calibration = str(calibration)
            output = Path(request.output_directory or self.args.output).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            destination = output / request.filename
            if request.kind == "motion" and destination.exists():
                raise HTTPException(409, "Output BVH exists. Choose another filename")
            take_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            folder = self.root / take_id
            folder.mkdir()
            take = dict(id=take_id, kind=request.kind, state="preparing", created=time.time(),
                        options=request.model_dump(), destination=str(destination), cameras=self.cameras.copy(),
                        transport=self.transport)
            self.takes[take_id] = take
            self.update(take)
            try:
                self.health_cache = (0, [])
                status = {info["camera"]: info for info in await self.health()}
                for name, info in status.items():
                    if info.get("camera") != name or not info.get("ready") or info.get("active"):
                        raise ValueError(f"{name}: {info.get('error') or 'node not ready, busy, or identity mismatch'}")
                self.update(take, camera_settings=status)
                self.capture = RTSPRecorder(self.cameras, folder, self.ffmpeg, self.ffprobe, self.capture_timeout)
                await self.capture.start()
                self.capture_watch = asyncio.create_task(self.watch_capture(take))
                message = ("All cameras recording. No flash required; begin movement or calibration."
                           if request.sync_method == "receive" else "All cameras recording. Fire the common flash now.")
                self.update(take, state="recording", message=message)
            except BaseException as exc:
                await self.stop_capture()
                self.update(take, state="failed", error=str(exc) or type(exc).__name__, capture_failed=True)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise HTTPException(503, str(exc)) from exc
            return take

    async def watch_capture(self, take):
        while take["state"] == "recording":
            await asyncio.sleep(.25)
            error = self.capture.failure()
            if error:
                async with self.lock:
                    if take["state"] != "recording":
                        return
                    await self.capture.stop()
                    self.update(take, state="failed", capture_failed=True, error=error)
                return

    async def stop_capture(self):
        if self.capture_watch:
            self.capture_watch.cancel()
            await asyncio.gather(self.capture_watch, return_exceptions=True)
            self.capture_watch = None
        if self.capture:
            await self.capture.stop()

    async def stop(self, take_id):
        async with self.lock:
            take = self.get(take_id)
            if take.get("capture_failed"):
                raise HTTPException(409, "RTSP recording failed. Restore the stream and make a new take; partial files are preserved")
            if take["cameras"] != self.cameras:
                raise HTTPException(409, "Restart with this recording's original --camera endpoints to recover it")
            if self.task and not self.task.done():
                raise HTTPException(409, "A collection/processing task is already active")
            if any(t["id"] != take_id and t["state"] in BUSY for t in self.takes.values()):
                raise HTTPException(409, "Finish the other active recording first")
            if take["state"] not in ("recording", "interrupted", "failed"):
                raise HTTPException(409, "Recording is not active or recoverable")
            self.update(take, state="collecting", error=None)
            self.task = asyncio.create_task(self.collect_and_process(take))
            return take

    async def collect(self, take):
        await self.stop_capture()
        if self.capture and self.capture.folder == self.root / take["id"]:
            if error := self.capture.failure():
                self.update(take, capture_failed=True)
                raise RuntimeError(error)
            await self.capture.finalize(self.cameras, take["id"])
            self.capture = None
        else:
            await finalize_recordings(self.root / take["id"], self.cameras, take["id"], self.ffprobe)
        self.health_cache = (0, [])

    async def collect_and_process(self, take):
        folder = self.root / take["id"]
        try:
            await self.collect(take)
            options = take["options"]
            import cv2
            if options["dictionary"] not in ("DICT_4X4_50", "DICT_4X4_250", "DICT_5X5_100", "DICT_6X6_250"):
                raise ValueError("Unsupported ChArUco dictionary")
            spec = dict(folder=str(folder), kind=take["kind"], calibration=options["calibration"],
                        sync_method=options.get("sync_method", "flash"),
                        camera_offsets_ms=options.get("camera_offsets_ms", {}),
                        max_skew_ms=options.get("max_skew_ms", 50), max_age_ms=options.get("max_age_ms", 500),
                        brightness_threshold=options["brightness_threshold"], blender=self.args.blender,
                        board=dict(squares_x=options["squares_x"], squares_y=options["squares_y"],
                                   square_length_mm=options["square_mm"], marker_length_ratio=options["marker_ratio"],
                                   aruco_dictionary_enum=getattr(cv2.aruco, options["dictionary"])))
            save(folder / "spec.json", spec)
            (folder / "result.json").unlink(missing_ok=True)
            (folder / "progress.json").unlink(missing_ok=True)
            self.update(take, state="processing")
            with (folder / "worker.log").open("w") as log:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", str(HERE / "freemocap_recording_worker.py"), str(folder / "spec.json"),
                    stdout=log, stderr=log, start_new_session=True)
                return_code = await self.process.wait()
            self.process = None
            if return_code:
                progress = json.loads((folder / "progress.json").read_text()) if (folder / "progress.json").exists() else {}
                raise RuntimeError(progress.get("detail", f"Worker exited {return_code}; see worker.log"))
            result = json.loads((folder / "result.json").read_text())
            save_error = None
            if "bvh" in result:
                destination = Path(take["destination"])
                try:
                    publish_bvh(Path(result["bvh"]), str(destination.parent), destination.name)
                    result["saved_bvh"] = str(destination)
                except OSError as exc:
                    save_error = f"BVH ready, but could not copy to selected location: {exc}. Choose another path and Save BVH."
            self.update(take, state="complete", result=result, error=save_error)
        except asyncio.CancelledError:
            self.update(take, state="interrupted", error="Host shutting down; files preserved")
            raise
        except Exception as exc:
            self.update(take, state="failed", error=f"{type(exc).__name__}: {exc}")

    async def close(self):
        await asyncio.gather(*(preview.close() for preview in self.previews.values()))
        if self.task and not self.task.done():
            self.task.cancel()
        if self.process and self.process.returncode is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 8)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
        await self.stop_capture()
        for take in self.takes.values():
            if take["state"] == "recording":
                self.update(take, state="interrupted", error="Host stopped; collect the finalized files after restart")


def create_app(args):
    host = Host(args)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await host.close()

    app = FastAPI(title="FreeMoCap recording host", lifespan=lifespan)
    app.state.host = host

    @app.get("/", response_class=HTMLResponse)
    async def page():
        return (HERE / "recording_ui.html").read_text()

    @app.get("/assets/{name}")
    async def asset(name: str):
        if name not in ("bvh_player.js", "armature_view.js"):
            raise HTTPException(404)
        return FileResponse(HERE / name, media_type="application/javascript")

    @app.get("/api/state")
    async def state():
        takes = []
        for take in sorted(host.takes.values(), key=lambda t: t["created"], reverse=True):
            view = dict(take)
            progress = host.root / take["id"] / "progress.json"
            if progress.is_file():
                view["progress"] = json.loads(progress.read_text())
            takes.append(view)
        return dict(cameras=await host.health(), takes=takes, output=str(args.output), records=str(host.root),
                    transport=host.transport)

    @app.get("/api/cameras/{name}/preview.jpg")
    async def preview(name: str):
        if name not in host.previews:
            raise HTTPException(404, "Unknown camera")
        source = host.previews[name]
        jpeg = source.snapshot()
        headers = {"Cache-Control": "no-store"}
        if jpeg is None:
            return Response(source.error or "Waiting for a fresh frame", status_code=503,
                            headers=headers, media_type="text/plain")
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @app.post("/api/recordings")
    async def start(request: StartRequest):
        return await host.start(request)

    @app.post("/api/recordings/{take_id}/stop")
    async def stop(take_id: str):
        return await host.stop(take_id)

    @app.get("/api/directories")
    async def directories(path: str = ""):
        folder = Path(path or args.output).expanduser().resolve()
        try:
            return dict(path=str(folder), parent=str(folder.parent),
                        directories=sorted(str(p) for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")))
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/recordings/{take_id}/bvh")
    async def bvh(take_id: str):
        take = host.get(take_id)
        path = take.get("result", {}).get("bvh")
        if not path or not Path(path).is_file():
            raise HTTPException(404, "BVH is not ready")
        return FileResponse(path, media_type="text/plain", filename=Path(take["destination"]).name)

    @app.post("/api/recordings/{take_id}/save")
    async def save_bvh(take_id: str, request: SaveRequest):
        take = host.get(take_id)
        source = take.get("result", {}).get("bvh")
        if not source:
            raise HTTPException(409, "BVH is not ready")
        try:
            path = publish_bvh(Path(source), request.output_directory, request.filename)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        host.update(take, destination=str(path), result={**take["result"], "saved_bvh": str(path)}, error=None)
        return dict(saved_bvh=str(path))

    @app.get("/api/recordings/{take_id}/log", response_class=HTMLResponse)
    async def log(take_id: str):
        import html
        host.get(take_id)
        path = host.root / take_id / "worker.log"
        text = path.read_bytes()[-64000:].decode(errors="replace") if path.exists() else "Worker has not started"
        for capture_log in sorted((host.root / take_id / "rtsp_capture").glob("*.log")):
            text += f"\n\n{capture_log.name}\n" + capture_log.read_bytes()[-16000:].decode(errors="replace")
        return "<meta charset=utf-8><pre>" + html.escape(text) + "</pre>"

    @app.get("/api/recordings/{take_id}/sync")
    async def sync(take_id: str):
        host.get(take_id)
        path = host.root / take_id / "sync_result.json"
        if not path.exists():
            raise HTTPException(404, "Synchronization is not ready")
        return json.loads(path.read_text())

    return app


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--camera", action="append", required=True,
                        help="cam1=rtsp://127.0.0.1:8554/cam1 (repeat for each camera)")
    result.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    result.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    result.add_argument("--capture-timeout", type=int, choices=range(3, 61), default=10,
                        metavar="SECONDS", help="RTSP connection / no-frame timeout, 3..60 (default 10)")
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=8080)
    result.add_argument("--records", type=Path, default=HERE.parent / "records" / "freemocap")
    result.add_argument("--output", type=Path, default=HERE.parent / "output")
    result.add_argument("--blender", default=shutil.which("blender") or "blender")
    return result


if __name__ == "__main__":
    import uvicorn
    os.environ["PATH"] = str(HERE / ".tools" / "bin") + os.pathsep + os.environ.get("PATH", "")
    args = parser().parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1)
