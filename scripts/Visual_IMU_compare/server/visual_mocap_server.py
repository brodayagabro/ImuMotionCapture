#!/usr/bin/env python3
"""Web controller: camera-local recordings → checked upload → native SkellySync audio synchronization."""

from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler
import uuid

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("recording-host")
BUSY = {"preparing", "recording", "stopping", "uploading", "synchronizing"}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def checked_name(name):
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[\w][\w .-]{0,79}", name)
        or ".." in name
    ):
        raise ValueError("Name: 1–80 letters/digits/spaces/_/./-, without .. or paths")
    return name


def cameras(values):
    result = {}
    for value in values:
        name, sep, url = value.partition("=")
        parsed = urlsplit(url)
        if (
            not sep
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name)
            or parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("Use --camera cam1=http://CAMERA_IP:8081")
        if name in result or url.rstrip("/") in result.values():
            raise ValueError("Camera names and URLs must be unique")
        result[name] = url.rstrip("/")
    if not 2 <= len(result) <= 8:
        raise ValueError("Configure 2–8 camera nodes")
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def synchronize(folder):
    """Use the pinned upstream audio pipeline; no pose/calibration/GPU imports."""
    from skelly_synchronize.core.models import SyncRequest, SyncMethod, VideoBackendKind
    from skelly_synchronize.core.pipeline.runner import SyncPipeline
    from skelly_synchronize.core.pipeline.stages import AudioLagStage, ProbeStage
    import numpy as np

    class CountedProbeStage(ProbeStage):
        def run(self, context, progress_callback):
            context = super().run(context, progress_callback)
            for video in context.videos:
                # OpenCV estimates MKV frame count from the *container* duration,
                # including any trailing audio. Our MJPEG/H.264 Matroska recordings
                # contain one encoded video frame per packet: count those without
                # decoding full-resolution video or replacing native synchronization.
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-count_packets",
                        "-show_entries",
                        "stream=codec_name,nb_read_packets",
                        "-of",
                        "json",
                        str(video.filepath),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3600,
                )
                stream = json.loads(result.stdout)["streams"][0]
                if stream["codec_name"] not in ("mjpeg", "h264"):
                    raise ValueError("Camera recording must contain MJPEG or H.264")
                video.frame_count = int(stream["nb_read_packets"])
                if video.frame_count < 12:
                    raise ValueError(f"{video.video_name}: fewer than 12 video frames")
            context.videos_before = list(context.videos)
            return context

    class CheckedAudioStage(AudioLagStage):
        def run(self, context, progress_callback):
            context = super().run(context, progress_callback)
            for name, samples in context.audio_signals.items():
                if (
                    not len(samples)
                    or not np.isfinite(samples).all()
                    or np.max(np.abs(samples)) < 1e-5
                ):
                    raise ValueError(
                        f"{name}: audio is empty/silent; use a common audible clap or speech"
                    )
            return context

    output = folder / "synchronized_videos"
    report = folder / "sync_result.json"
    if report.is_file():
        return json.loads(report.read_text())
    if output.exists():
        # Preserve outputs of an interrupted previous attempt for diagnosis.
        output.rename(folder / ("incomplete-sync-" + uuid.uuid4().hex[:8]))
    request = SyncRequest(
        raw_video_folder_path=folder / "raw_videos",
        synchronized_video_folder_path=output,
        method=SyncMethod.AUDIO,
        video_handler=VideoBackendKind.FFMPEG,
        create_debug_artifacts=True,
    )
    pipeline = SyncPipeline.for_request(request)
    pipeline.stages = [
        CheckedAudioStage()
        if isinstance(stage, AudioLagStage)
        else CountedProbeStage()
        if isinstance(stage, ProbeStage)
        else stage
        for stage in pipeline.stages
    ]
    result = pipeline.run(
        request,
        progress_callback=lambda name, fraction: save(
            folder / "progress.json", dict(stage=name, fraction=fraction)
        ),
    )
    if result.synchronized_frame_count < 12:
        raise ValueError(
            "Less than 12 synchronized frames; record longer overlapping videos"
        )
    value = result.model_dump(mode="json")
    value.update(
        method="skellysync_audio",
        package_commit="0231c41a3bcb967e231163943a39dd40b50278f1",
    )
    save(report, value)
    return value


class Host:
    def __init__(self, args):
        self.args = args
        self.cameras = cameras(args.camera)
        self.root = args.records.expanduser().resolve()
        (self.root / "takes").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.worker = None
        self.process = None
        self.opener = build_opener(ProxyHandler({}))
        self.takes = {}
        self.health = {}
        self.preview_cache = {}
        self.preview_locks = {name: threading.Lock() for name in self.cameras}
        for path in (self.root / "takes").glob("*/take.json"):
            take = json.loads(path.read_text())
            if take["state"] in BUSY:
                take.update(
                    state="interrupted",
                    error="Host restarted; use Upload / retry to finish this take",
                )
                save(path, take)
            self.takes[take["id"]] = take
        self.calibration_path = self.root / "current_calibration.json"
        self.calibration = (
            json.loads(self.calibration_path.read_text())
            if self.calibration_path.exists()
            else None
        )
        self.poller = threading.Thread(target=self.poll, daemon=True)
        self.poller.start()

    def request(self, name, path, body=None, timeout=20, headers=None):
        options = dict(headers or {})
        if self.args.camera_token:
            options["Authorization"] = "Bearer " + self.args.camera_token
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            options["Content-Type"] = "application/json"
        request = Request(self.cameras[name] + path, data=data, headers=options)
        return self.opener.open(request, timeout=timeout)

    def json_request(self, name, path, body=None, timeout=20):
        try:
            with self.request(name, path, body, timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.load(exc).get("error", str(exc))
            except Exception:
                detail = str(exc)
            raise ValueError(f"{name}: {detail}") from exc

    def poll(self):
        def status(name):
            try:
                info = self.json_request(name, "/api/status", timeout=3)
                if info.get("camera") != name:
                    raise ValueError("Camera name differs from configured ID")
                return dict(name=name, online=True, **info)
            except Exception as exc:
                return dict(name=name, online=False, error=str(exc))

        with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
            while not self.closed.is_set():
                values = list(pool.map(status, self.cameras))
                with self.lock:
                    self.health = {v["name"]: v for v in values}
                self.closed.wait(2)

    def update(self, take, **values):
        with self.lock:
            take.update(values)
            save(self.root / "takes" / take["id"] / "take.json", take)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(
                dict(
                    cameras=[
                        self.health.get(
                            n, dict(name=n, online=False, error="Connecting…")
                        )
                        for n in self.cameras
                    ],
                    takes=sorted(
                        self.takes.values(), key=lambda t: t["created"], reverse=True
                    ),
                    calibration=self.calibration,
                    records=str(self.root),
                )
            )

    def launch(self, func, *args):
        if self.worker and self.worker.is_alive():
            raise ValueError("An operation is already running")
        self.worker = threading.Thread(target=func, args=args, daemon=True)
        self.worker.start()

    def start(self, body):
        name = checked_name(body.get("name"))
        kind = body.get("kind")
        if kind not in ("calibration", "record"):
            raise ValueError("kind must be calibration or record")
        with self.lock:
            if self.closed.is_set() or (self.worker and self.worker.is_alive()):
                raise ValueError(
                    "An operation is already running or host is shutting down"
                )
            if any(t["state"] in BUSY for t in self.takes.values()):
                raise ValueError("A recording or transfer is already active")
            if kind == "calibration" and self.calibration:
                if (
                    not body.get("replace_calibration")
                    or body.get("expected_calibration_id") != self.calibration["id"]
                ):
                    raise ValueError(
                        "Calibration already exists. Confirm replacing the current calibration set"
                    )
            identifier = uuid.uuid4().hex
            folder = self.root / "takes" / identifier
            folder.mkdir()
            take = dict(
                id=identifier,
                name=name,
                kind=kind,
                state="preparing",
                created=time.time(),
                cameras=self.cameras.copy(),
                calibration_id=(self.calibration or {}).get("id")
                if kind == "record"
                else None,
            )
            self.takes[identifier] = take
            self.update(take)
            self.launch(self.prepare, take)
            return dict(take)

    def prepare(self, take):
        try:
            body = {key: take[key] for key in ("id", "name", "kind")}
            with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
                futures = {
                    name: pool.submit(self.json_request, name, "/api/start", body, 25)
                    for name in self.cameras
                }
                errors = []
                for name, future in futures.items():
                    try:
                        result = future.result()
                        if (
                            result.get("state") != "recording"
                            or result.get("camera") != name
                        ):
                            raise ValueError("Camera is not recording")
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")
            if errors:
                raise ValueError("; ".join(errors))
            if self.closed.is_set():
                raise InterruptedError("Host shutting down")
            self.update(take, state="recording", error=None)
        except Exception as exc:
            for name in self.cameras:
                try:
                    self.json_request(name, "/api/stop", {"id": take["id"]}, timeout=5)
                except Exception:
                    pass
            self.update(
                take,
                state="interrupted" if self.closed.is_set() else "failed",
                error=str(exc),
            )

    def stop(self, identifier, upload=True):
        with self.lock:
            take = self.takes[identifier]
            if take["state"] not in ("recording", "stopped", "failed", "interrupted"):
                raise ValueError("Take is not available for stopping/upload")
            if take["cameras"] != self.cameras:
                raise ValueError("Restore the camera configuration used by this take")
            if any(
                t["state"] in BUSY and t["id"] != identifier
                for t in self.takes.values()
            ):
                raise ValueError("Another operation is active")
            self.launch(self.collect, take, upload)
            return dict(take)

    def collect(self, take, upload):
        try:
            self.update(take, state="stopping", error=None)
            with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
                list(
                    pool.map(
                        lambda name: self.json_request(
                            name, "/api/stop", {"id": take["id"]}, timeout=20
                        ),
                        self.cameras,
                    )
                )
            deadline = time.monotonic() + 600
            manifests = {}
            while len(manifests) < len(self.cameras):
                if self.closed.wait(0.5):
                    raise InterruptedError(
                        "Host shutting down; transfer can be resumed"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Cameras did not finalize recordings in 10 minutes"
                    )
                for name in self.cameras:
                    if name in manifests:
                        continue
                    value = self.json_request(
                        name, "/api/recordings/" + take["id"], timeout=10
                    )
                    if value["state"] == "failed":
                        raise ValueError(f"{name}: {value.get('error')}")
                    if value["state"] == "ready":
                        if (
                            any(
                                value.get(key) != take[key]
                                for key in ("id", "name", "kind")
                            )
                            or value.get("camera") != name
                        ):
                            raise ValueError(f"{name}: manifest identity mismatch")
                        if (
                            type(value.get("bytes")) is not int
                            or not 0 < value["bytes"] <= 1024**4
                            or not re.fullmatch("[a-f0-9]{64}", value.get("sha256", ""))
                        ):
                            raise ValueError(f"{name}: invalid file manifest")
                        manifests[name] = value
            folder = self.root / "takes" / take["id"]
            manifest_path = folder / "camera_manifests.json"
            if (
                manifest_path.exists()
                and json.loads(manifest_path.read_text()) != manifests
            ):
                raise ValueError(
                    "Camera manifests changed; refusing to resume different files"
                )
            save(manifest_path, manifests)
            self.update(
                take,
                state="stopped",
                total_bytes=sum(v["bytes"] for v in manifests.values()),
            )
            if not upload:
                return
            raw = folder / "raw_videos"
            raw.mkdir(exist_ok=True)
            self.update(take, state="uploading", transferred_bytes=0)
            for name, manifest in manifests.items():
                self.download(
                    take, name, manifest, raw / f"{take['name']}.id-{name}.idx-0.mkv"
                )
            self.update(
                take, state="synchronizing", transferred_bytes=take["total_bytes"]
            )
            with (folder / "sync.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--sync-job",
                        str(folder),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                while self.process.poll() is None:
                    if self.closed.wait(0.2):
                        raise InterruptedError(
                            "Host shutting down; synchronization can be retried"
                        )
                    progress = folder / "progress.json"
                    if progress.exists():
                        try:
                            with self.lock:
                                take["progress"] = json.loads(progress.read_text())
                        except (OSError, ValueError):
                            pass
                if self.process.returncode:
                    raise RuntimeError(
                        "SkellySync failed: "
                        + (folder / "sync.log").read_text(errors="replace")[-1800:]
                    )
            result = json.loads((folder / "sync_result.json").read_text())
            destination = str(folder / "synchronized_videos")
            self.update(
                take,
                state="ready",
                error=None,
                synchronized_path=destination,
                sync=result,
            )
            if take["kind"] == "calibration":
                with self.lock:
                    self.calibration = dict(
                        id=take["id"], name=take["name"], synchronized_path=destination
                    )
                    save(self.calibration_path, self.calibration)
        except InterruptedError as exc:
            self.update(take, state="interrupted", error=str(exc))
        except Exception as exc:
            self.update(take, state="failed", error=str(exc))

    def download(self, take, name, manifest, target):
        if target.exists():
            if (
                target.stat().st_size != manifest["bytes"]
                or digest(target) != manifest["sha256"]
            ):
                raise ValueError(
                    f"{name}: existing received file does not match its manifest"
                )
            return
        partial = target.with_suffix(".partial")
        for attempt in range(3):
            if self.closed.is_set():
                raise InterruptedError("Upload interrupted; partial file preserved")
            start = partial.stat().st_size if partial.exists() else 0
            if start >= manifest["bytes"]:
                if start == manifest["bytes"] and digest(partial) == manifest["sha256"]:
                    partial.replace(target)
                    return
                partial.unlink()
                start = 0
            if (
                shutil.disk_usage(target.parent).free
                < manifest["bytes"] - start + 256 * 1024**2
            ):
                raise ValueError("Not enough free space on the host")
            try:
                with self.request(
                    name,
                    "/files/" + take["id"],
                    timeout=30,
                    headers={"Range": f"bytes={start}-"} if start else {},
                ) as response:
                    if start and (
                        response.status != 206
                        or not response.headers.get("Content-Range", "").startswith(
                            f"bytes {start}-"
                        )
                    ):
                        raise ValueError("Camera did not honor resume range")
                    if (
                        int(response.headers.get("Content-Length", "-1"))
                        != manifest["bytes"] - start
                    ):
                        raise ValueError("Incorrect upload size")
                    with partial.open("ab" if start else "wb") as stream:
                        written = start
                        updated = 0
                        while block := response.read(256 * 1024):
                            if self.closed.is_set():
                                raise InterruptedError(
                                    "Upload interrupted; partial file preserved"
                                )
                            written += len(block)
                            if written > manifest["bytes"]:
                                raise ValueError("Camera sent more bytes than declared")
                            stream.write(block)
                            if time.monotonic() - updated > 1:
                                updated = time.monotonic()
                                self.update(
                                    take,
                                    transfer_camera=name,
                                    transferred_bytes=sum(
                                        p.stat().st_size
                                        for p in target.parent.iterdir()
                                        if p.is_file()
                                    ),
                                )
                        stream.flush()
                        os.fsync(stream.fileno())
                if partial.stat().st_size != manifest["bytes"]:
                    raise ValueError("Incomplete upload")
                if digest(partial) != manifest["sha256"]:
                    partial.unlink()
                    raise ValueError("Upload SHA-256 mismatch")
                partial.replace(target)
                return
            except InterruptedError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                if self.closed.wait(attempt + 1):
                    raise InterruptedError("Upload interrupted; partial file preserved")

    def preview(self, name):
        if name not in self.cameras:
            raise KeyError(name)
        with self.preview_locks[name]:
            cached = self.preview_cache.get(name)
            if cached and time.monotonic() - cached[0] < 1:
                return cached[1]
            with self.request(name, "/preview.jpg", timeout=6) as response:
                data = response.read(512 * 1024 + 1)
                if len(data) > 512 * 1024 or not data.startswith(b"\xff\xd8"):
                    raise ValueError("Invalid preview image")
            self.preview_cache[name] = (time.monotonic(), data)
            return data

    def close(self):
        self.closed.set()
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        for take in list(self.takes.values()):
            if take["state"] in ("preparing", "recording"):
                for name in self.cameras:
                    try:
                        self.json_request(
                            name, "/api/stop", {"id": take["id"]}, timeout=3
                        )
                    except Exception:
                        pass
                self.update(
                    take,
                    state="interrupted",
                    error="Host stopped; cameras were asked to finalize recordings",
                )
        if self.worker:
            self.worker.join(timeout=35)
        self.poller.join(timeout=5)


def handler(host):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(
            self, data, content_type="application/json; charset=utf-8", status=200
        ):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self.send(HTML.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.send(host.snapshot())
                elif path.startswith("/preview/"):
                    try:
                        self.send(
                            host.preview(
                                path.removeprefix("/preview/").removesuffix(".jpg")
                            ),
                            "image/jpeg",
                        )
                    except KeyError:
                        raise
                    except Exception:
                        self.send({"error": "Preview unavailable"}, status=503)
                elif path.startswith("/api/takes/"):
                    identifier = path.rsplit("/", 1)[-1]
                    with host.lock:
                        value = copy.deepcopy(host.takes[identifier])
                    self.send(value)
                else:
                    self.send({"error": "Unknown route"}, status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except KeyError:
                self.send({"error": "Unknown recording/camera"}, status=404)
            except Exception as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            try:
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("application/json required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid JSON size")
                body = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == "/api/start":
                    value = host.start(body)
                elif path == "/api/stop":
                    value = host.stop(body["id"], bool(body.get("upload", True)))
                elif path == "/api/upload":
                    value = host.stop(body["id"], True)
                else:
                    self.send({"error": "Unknown route"}, status=404)
                    return
                self.send(value, status=202)
            except KeyError:
                self.send(
                    {"error": "Unknown recording or required field missing"}, status=404
                )
            except Exception as exc:
                self.send({"error": str(exc)}, status=409)

    return Handler


HTML = r"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Запись камер</title><style>
*{box-sizing:border-box}[hidden]{display:none!important}body{font:16px system-ui;background:#f3f6fa;color:#203146;margin:24px auto;max-width:1200px;padding:0 16px}h1{font-size:26px}section,.cam{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:18px;margin:16px 0}button,input,select{font:inherit;border:1px solid #bccbdc;border-radius:6px;padding:8px;max-width:100%}button{background:#eaf2fb;cursor:pointer}button:disabled{opacity:.5;cursor:wait}label{display:inline-block;margin:6px 14px 6px 0}input[type=text]{width:260px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.cam{margin:0}.cam img{width:100%;max-width:320px;display:block;margin:10px auto}.muted{color:#63768a;font-size:14px}#error,.bad{color:#ac263a;white-space:pre-wrap}.take{padding:12px 0;border-top:1px solid #e0e7ef;overflow-wrap:anywhere}.take button{margin:8px 8px 0 0}.path{font:13px monospace;overflow-wrap:anywhere}.good{color:#24704c}@media(max-width:600px){body{margin:12px auto}label{display:block}}
</style><h1>Запись камер → синхронизация</h1><p class="muted">Видео и звук записываются на камерах. После выгрузки SkellySync выравнивает файлы по общему звуку.</p>
<section><div id="calibration">Калибровочный набор пока не записан</div><label>Имя <input id="name" type="text" maxlength="80" value="take_001"></label><label>Тип <select id="kind"><option value="record">record — движение</option><option value="calibration">calibration — доска</option></select></label>
<p><button id="start">Начать запись на всех камерах</button> <label><input id="autoUpload" type="checkbox" checked> Выгрузить и синхронизировать после остановки</label></p><p class="muted">Когда все камеры записывают, сделайте общий хлопок или произнесите несколько слов. Затем выполняйте движение или показывайте калибровочную доску.</p><div id="error" role="alert"></div></section>
<label><input id="preview" type="checkbox" checked> Превью камер — до 320 px, 1 кадр/с</label><div class="grid" id="cameras"></div>
<section><h2>Записи</h2><div id="takes">Подключение…</div></section>
<script>
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={preparing:'Подготовка камер',recording:'Все камеры записывают',stopping:'Завершение файлов на камерах',stopped:'Готово к выгрузке',uploading:'Выгрузка на сервер',synchronizing:'Синхронизация SkellySync',ready:'Готово для официального FreeMoCap',failed:'Ошибка',interrupted:'Прервано'};
let state=null,polling=false,lastHTML='';const cards=new Map();
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const value=await r.json();if(!r.ok)throw Error(value.error||r.status);return value}
function fail(e){$('error').textContent=e.message||String(e)}
async function preview(name,card){if(document.hidden||!$('preview').checked){card.img.hidden=true}else{try{const r=await fetch('/preview/'+encodeURIComponent(name)+'.jpg',{signal:AbortSignal.timeout(7000)});if(!r.ok)throw Error('Превью недоступно');const url=URL.createObjectURL(await r.blob());card.img.src=url;card.img.hidden=false;if(card.url)URL.revokeObjectURL(card.url);card.url=url;card.note.textContent=''}catch(e){card.img.hidden=true;card.note.textContent=e.message}}setTimeout(()=>preview(name,card),1000)}
async function refresh(){if(polling)return;polling=true;try{state=await api('/api/state');$('calibration').textContent=state.calibration?'Текущий калибровочный набор: '+state.calibration.name:'Калибровочный набор пока не записан';$('start').disabled=state.takes.some(t=>['preparing','recording','stopping','uploading','synchronizing'].includes(t.state));
 for(const c of state.cameras){let card=cards.get(c.name);if(!card){const box=document.createElement('div');box.className='cam';const title=document.createElement('b'),info=document.createElement('p'),img=document.createElement('img'),note=document.createElement('small');title.textContent=c.name;info.className='muted';img.alt=c.name;img.hidden=true;box.append(title,info,img,note);$('cameras').append(box);card={info,img,note};cards.set(c.name,card);preview(c.name,card)}card.info.textContent=c.online?`${c.mode.width}×${c.mode.height} · ${c.mode.fps} FPS · ${c.mode.pixel_format} · свободно ${(c.free_bytes/1024**3).toFixed(1)} ГБ · ${c.active_state==='recording'?'записывает':c.active_state==='stopping'?'завершает файл':'ожидает'}${c.synthetic?' · ТЕСТОВОЕ ВИДЕО':''}`:c.error}
 const html=state.takes.map(t=>`<div class="take"><b>${esc(t.name)}</b> · ${esc(t.kind)}<p>${esc(labels[t.state]||t.state)}${t.state==='uploading'?' · '+Math.min(100,100*(t.transferred_bytes||0)/t.total_bytes).toFixed(0)+'%':''}</p>${t.error?`<div class="bad">${esc(t.error)}</div>`:''}${t.state==='recording'?`<button data-stop="${t.id}">Остановить запись</button>`:''}${['stopped','failed','interrupted'].includes(t.state)?`<button data-upload="${t.id}">Выгрузить / повторить</button>`:''}${t.synchronized_path?`<div class="path">${esc(t.synchronized_path)}</div><button data-copy="${t.id}">Копировать путь</button><a href="/api/takes/${t.id}" target="_blank">Отчёт</a>`:''}</div>`).join('')||'Записей пока нет';if(html!==lastHTML){$('takes').innerHTML=html;lastHTML=html}
 }catch(e){fail(e)}finally{polling=false}}
$('start').onclick=async()=>{try{$('error').textContent='';let replace=false;const kind=$('kind').value;if(kind==='calibration'&&state?.calibration){if(!confirm('Калибровочный набор «'+state.calibration.name+'» уже есть. Заменить его новым после успешной выгрузки? Предыдущие файлы сохранятся.'))return;replace=true}$('start').disabled=true;await api('/api/start',{name:$('name').value.trim(),kind,replace_calibration:replace,expected_calibration_id:state?.calibration?.id});await refresh()}catch(e){fail(e);$('start').disabled=false}};
$('takes').onclick=async e=>{const b=e.target.closest('button');if(!b)return;try{$('error').textContent='';if(b.dataset.stop){b.disabled=true;await api('/api/stop',{id:b.dataset.stop,upload:$('autoUpload').checked})}if(b.dataset.upload){b.disabled=true;await api('/api/upload',{id:b.dataset.upload})}if(b.dataset.copy){const path=state.takes.find(t=>t.id===b.dataset.copy).synchronized_path;if(navigator.clipboard)await navigator.clipboard.writeText(path);else prompt('Путь к синхронизированным видео',path)}await refresh()}catch(e){b.disabled=false;fail(e)}};
refresh();setInterval(refresh,2000);
</script></html>"""


def main():
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="cam1=http://CAMERA_IP:8081; repeat for every camera",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--records", type=Path, default=HERE.parent / "records" / "local_capture"
    )
    parser.add_argument("--camera-token-file", type=Path)
    parser.add_argument("--sync-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sync_job:
        synchronize(args.sync_job.resolve())
        return
    args.camera_token = (
        args.camera_token_file.read_text().strip() if args.camera_token_file else ""
    )
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    host = Host(args)
    http = ThreadingHTTPServer((args.host, args.port), handler(host))
    http.daemon_threads = True

    def stop(*_):
        threading.Thread(target=http.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Open http://SERVER_IP:{args.port}; recordings: {host.root}", flush=True)
    try:
        http.serve_forever()
    finally:
        http.server_close()
        host.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
