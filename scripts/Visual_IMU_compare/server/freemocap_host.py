#!/usr/bin/env python3
"""Distributed SkellyCam recording → native offline FreeMoCap → BVH + web player."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time
from urllib.parse import quote, urlsplit
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

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
            raise ValueError("Use --camera cam1=http://CAMERA_IP:8082")
        if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("A camera endpoint must be an HTTP origin")
        if name in endpoints:
            raise ValueError(f"Duplicate camera identity: {name}")
        endpoints[name] = url.rstrip("/")
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
        self.takes = {}
        self.lock = asyncio.Lock()
        self.task = None
        self.process = None
        self.client = None
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

    async def node(self, name, method, path):
        response = await self.client.request(method, self.cameras[name] + path)
        if response.is_error:
            raise RuntimeError(f"{name}: HTTP {response.status_code}: {response.text[:1000]}")
        return response.json()

    async def all_nodes(self, method, path):
        results = await asyncio.gather(*(self.node(name, method, path) for name in self.cameras), return_exceptions=True)
        errors = [f"{name}: {value}" for name, value in zip(self.cameras, results) if isinstance(value, BaseException)]
        if errors:
            raise RuntimeError("; ".join(errors))
        return dict(zip(self.cameras, results))

    async def health(self):
        async def one(name):
            try:
                value = await asyncio.wait_for(self.node(name, "GET", "/health"), 3)
                if value.get("camera") != name or value.get("engine") != "skellycam":
                    raise ValueError("Camera identity / engine mismatch")
                return value
            except Exception as exc:
                return dict(camera=name, error=str(exc))
        return await asyncio.gather(*(one(name) for name in self.cameras))

    async def start(self, request):
        async with self.lock:
            if any(t["state"] in BUSY for t in self.takes.values()):
                raise HTTPException(409, "A recording or processing task is already active")
            if request.kind not in ("sync", "calibration", "motion"):
                raise HTTPException(400, "Unknown recording kind")
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
                        options=request.model_dump(), destination=str(destination), cameras=self.cameras.copy())
            self.takes[take_id] = take
            self.update(take)
            try:
                status = await self.all_nodes("POST", "/prepare")
                for name, info in status.items():
                    if info.get("camera") != name or not info.get("ready") or info.get("active"):
                        raise ValueError(f"{name}: node not ready, busy, or identity mismatch")
                rates = [info["fps"] for info in status.values()]
                if max(rates) - min(rates) > 0.01:
                    raise ValueError("Configure the same FPS on all camera nodes")
                self.update(take, camera_settings=status)
                await self.all_nodes("POST", f"/recordings/{take_id}/start")
                self.update(take, state="recording", message="All nodes recording. Fire the common flash now.")
            except Exception as exc:
                # Even a request that timed out may have started a camera.
                stops = await asyncio.gather(*(self.node(n, "POST", f"/recordings/{take_id}/stop") for n in self.cameras), return_exceptions=True)
                self.update(take, state="failed", error=str(exc), stop_results=[str(s) for s in stops])
                raise HTTPException(503, str(exc)) from exc
            return take

    async def stop(self, take_id):
        async with self.lock:
            take = self.get(take_id)
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
        await self.all_nodes("POST", f"/recordings/{take['id']}/stop")
        folder = self.root / take["id"]
        raw = folder / "raw_videos"
        raw.mkdir(exist_ok=True)
        manifests = await self.all_nodes("GET", f"/recordings/{take['id']}/manifest")
        for name, manifest in manifests.items():
            if manifest.get("camera") != name:
                raise ValueError(f"{name}: transfer identity mismatch")
            video_count = 0
            for item in manifest["files"]:
                relative = Path(item["path"])
                target = (folder / "camera_files" / name / relative).resolve()
                base = (folder / "camera_files" / name).resolve()
                if relative.is_absolute() or not target.is_relative_to(base):
                    raise ValueError("Unsafe path in camera manifest")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                digest = hashlib.sha256()
                size = 0
                url = self.cameras[name] + f"/recordings/{take['id']}/files/" + quote(relative.as_posix())
                async with self.client.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        async for chunk in response.aiter_bytes():
                            stream.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                if size != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Transfer checksum failed: {name}/{relative}")
                partial.replace(target)
                if relative.parent.name == "synchronized_videos" and relative.suffix == ".mp4":
                    video_count += 1
                    # Canonical native filenames preserve identity between calibration and motion.
                    shutil.copyfile(target, raw / f"{take['id']}.id-{name}.idx-0.mp4")
            if video_count != 1:
                raise ValueError(f"{name}: expected one native video, found {video_count}")
        save(folder / "transfer_manifests.json", manifests)

    async def collect_and_process(self, take):
        folder = self.root / take["id"]
        try:
            await self.collect(take)
            options = take["options"]
            import cv2
            if options["dictionary"] not in ("DICT_4X4_50", "DICT_4X4_250", "DICT_5X5_100", "DICT_6X6_250"):
                raise ValueError("Unsupported ChArUco dictionary")
            spec = dict(folder=str(folder), kind=take["kind"], calibration=options["calibration"],
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
        for take in self.takes.values():
            if take["state"] == "recording":
                await asyncio.gather(*(self.node(n, "POST", f"/recordings/{take['id']}/stop") for n in self.cameras), return_exceptions=True)
                self.update(take, state="interrupted", error="Host stopped; collect the finalized files after restart")


def create_app(args):
    host = Host(args)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False) as client:
            host.client = client
            yield
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
        return dict(cameras=await host.health(), takes=takes, output=str(args.output), records=str(host.root))

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
    result.add_argument("--camera", action="append", required=True, help="cam1=http://192.168.1.98:8082 (repeat)")
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
