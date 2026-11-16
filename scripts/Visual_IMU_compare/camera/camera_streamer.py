#!/usr/bin/env python3
"""Local camera recorder and HTTP control node. Python standard library + FFmpeg only."""

from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

LOG = logging.getLogger("camera-node")
PIXEL_FORMATS = {"MJPG": "mjpeg", "H264": "h264", "YUYV": "yuyv422", "NV12": "nv12"}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def checked_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid recording ID")
    return value


def checked_name(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[\w][\w .-]{0,79}", value)
        or ".." in value
    ):
        raise ValueError(
            "Name: 1–80 letters/digits/spaces/_/./-, without .. or path separators"
        )
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path, ffprobe):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    data = json.loads(result.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    duration = float(data.get("format", {}).get("duration", 0))
    if not audio or duration < 0.5:
        raise ValueError(
            "Recording needs a readable video and audio track, at least 0.5 seconds"
        )
    return dict(
        width=video["width"],
        height=video["height"],
        video_codec=video["codec_name"],
        audio_codec=audio["codec_name"],
        duration_s=duration,
        fps=float(Fraction(video["avg_frame_rate"])),
    )


@dataclass(frozen=True)
class Mode:
    pixel_format: str
    width: int
    height: int
    fps: float


def resolution(value):
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, e.g. 1280x720")
    width, height = map(int, match.groups())
    if min(width, height) < 16 or max(width, height) > 8192 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError(
            "H.264 dimensions must be even, between 16 and 8192"
        )
    return width, height


def rate(value):
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(
            "FPS must be a number or fraction such as 30000/1001"
        ) from None
    if not 0 < fps <= 240:
        raise argparse.ArgumentTypeError("FPS must be > 0 and <= 240")
    return fps


def parse_modes(text):
    """Discrete V4L2 mode enumeration; unsupported/stepwise modes stay explicit."""
    modes, pixel_format, size = [], None, None
    for line in text.splitlines():
        found = re.search(r"\[\d+\]:\s*'([^']+)'", line)
        if found:
            pixel_format = PIXEL_FORMATS.get(found[1])
            size = None
        found = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if found:
            size = tuple(map(int, found.groups()))
        elif "Size:" in line:
            size = None
        found = re.search(r"Interval:\s*Discrete.*\(([\d.]+)\s+fps\)", line)
        if found and pixel_format and size:
            modes.append(Mode(pixel_format, *size, float(found[1])))
    return modes


def probe_device(tool, device):
    result = subprocess.run(
        [tool, "--device", str(device), "--list-formats-ext"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode:
        raise ValueError(
            f"Cannot query {device}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return parse_modes(result.stdout), result.stdout


def discover_devices(tool):
    cameras = []
    for path in sorted(Path("/dev").glob("video[0-9]*"), key=lambda p: int(p.name[5:])):
        try:
            modes, text = probe_device(tool, path)
        except (ValueError, subprocess.TimeoutExpired):
            continue
        if modes:
            name_file = Path("/sys/class/video4linux") / path.name / "name"
            name = name_file.read_text().strip() if name_file.exists() else path.name
            cameras.append((path, name, modes, text))
    return cameras


def choose_device(args, tool):
    if args.device:
        path = (
            Path("/dev/video" + args.device)
            if args.device.isdecimal()
            else Path(args.device)
        )
        modes, text = probe_device(tool, path)
        return path, modes, text
    cameras = discover_devices(tool)
    if len(cameras) != 1:
        available = ", ".join(str(c[0]) for c in cameras) or "none"
        raise ValueError(
            f"Specify --device; usable cameras: {available}. Check --list-devices and device permissions."
        )
    path, _, modes, text = cameras[0]
    return path, modes, text


def select_mode(modes, size="max", fps=None, pixel_format="auto"):
    allowed = {"mjpeg", "h264"} if pixel_format == "auto" else {pixel_format}
    candidates = [
        m
        for m in modes
        if m.pixel_format in allowed
        and (size == "max" or (m.width, m.height) == size)
        and (
            fps is None or math.isclose(m.fps, float(fps), rel_tol=1e-4, abs_tol=0.001)
        )
    ]
    if not candidates:
        raise ValueError(
            "No matching compressed camera mode; use --list-formats and select MJPEG or H.264"
        )
    return max(
        candidates, key=lambda m: (m.width * m.height, m.fps, m.pixel_format == "h264")
    )


def command(args, mode, destination=None):
    cmd = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostats",
        "-stats_period",
        "0.5",
        "-progress",
        "pipe:2",
    ]
    if args.test_video:
        cmd += ["-re", "-stream_loop", "-1", "-i", str(args.test_video)]
    else:
        if destination:
            cmd += ["-copyts", "-start_at_zero"]
        if mode.pixel_format == "h264":
            cmd += ["-skip_frame", "nokey"]
        cmd += [
            "-thread_queue_size",
            "8",
            "-f",
            "v4l2",
            "-input_format",
            mode.pixel_format,
            "-timestamps",
            "abs",
            "-video_size",
            f"{mode.width}x{mode.height}",
            "-framerate",
            str(mode.fps),
            "-i",
            str(args.device),
        ]
        if destination:
            cmd += [
                "-isync",
                "0",
                "-thread_queue_size",
                "64",
                "-f",
                "alsa",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-i",
                args.audio,
            ]
    if destination:
        audio = "0:a:0" if args.test_video else "1:a:0"
        cmd += [
            "-map",
            "0:v:0",
            "-map",
            audio,
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            "-af",
            "aresample=async=1:first_pts=0",
            "-avoid_negative_ts",
            "make_zero",
            "-f",
            "matroska",
            "-n",
            str(destination),
        ]
    cmd += ["-map", "0:v:0", "-an"]
    if mode.pixel_format == "mjpeg":
        cmd += ["-c:v", "copy"]
    else:
        # Decode only keyframes for native H.264 cameras; recorded packets are copied.
        cmd += [
            "-vf",
            f"fps=1,scale=w='min(iw,{args.preview_width})':h=-2",
            "-c:v",
            "mjpeg",
            "-threads",
            "1",
            "-q:v",
            "10",
        ]
    return cmd + ["-f", "image2pipe", "pipe:1"]


class Capture:
    def __init__(self, args, mode, destination=None):
        self.args, self.mode = args, mode
        self.process = subprocess.Popen(
            command(args, mode, destination),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.jpeg = None
        self.updated = 0
        self.progress = time.monotonic()
        self.media_s = 0
        self.error = None
        self.stopping = False
        self.tail = []
        self.readers = [
            threading.Thread(target=f, daemon=True)
            for f in (self.read_images, self.read_progress)
        ]
        for reader in self.readers:
            reader.start()

    def read_images(self):
        pending = bytearray()
        try:
            while block := self.process.stdout.read1(65536):
                pending.extend(block)
                while pending:
                    start = pending.find(b"\xff\xd8")
                    if start < 0:
                        pending[:] = pending[-1:]
                        break
                    if start:
                        del pending[:start]
                    end = pending.find(b"\xff\xd9", 2)
                    if end < 0:
                        if len(pending) > 32 * 1024 * 1024:
                            pending.clear()
                        break
                    # Keep one compressed sample. No image decoding in Python.
                    now = time.monotonic()
                    if now - self.updated >= 1 / self.args.preview_fps:
                        self.jpeg = bytes(pending[: end + 2])
                        self.updated = now
                    del pending[: end + 2]
        except OSError as exc:
            self.error = str(exc)

    def read_progress(self):
        for raw in self.process.stderr:
            text = raw.decode(errors="replace").strip()
            key, _, value = text.partition("=")
            if key == "out_time_us":
                try:
                    seconds = int(value) / 1e6
                    if seconds > self.media_s:
                        self.media_s = seconds
                        self.progress = time.monotonic()
                except ValueError:
                    pass
            elif "=" not in text or text.startswith("["):
                self.tail = (self.tail + [text])[-15:]
        if not self.stopping:
            self.error = "FFmpeg ended: " + "; ".join(self.tail)[-1500:]

    def ready(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.error or self.process.poll() is not None:
                raise RuntimeError(self.error or "; ".join(self.tail))
            if self.jpeg and self.media_s > 0:
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Camera did not produce frames within 15 seconds: " + "; ".join(self.tail)
        )

    def finish(self):
        self.stopping = True
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b"q\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.error = "FFmpeg failed to finalize the recording"
        for reader in self.readers:
            reader.join(timeout=5)
        if self.process.returncode and not self.error:
            self.error = "FFmpeg failed: " + "; ".join(self.tail)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            pipe.close()


class Node:
    def __init__(self, args, mode, idle_mode):
        self.args, self.mode, self.idle_mode = args, mode, idle_mode
        self.root = args.records.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.preview_lock = threading.Lock()
        self.capture = None
        self.active = None
        self.finalizer = None
        self.preview_requested = 0
        self.thumbnail = None
        self.thumbnail_time = 0
        self.closed = threading.Event()
        self.takes = {}
        for path in self.root.glob("*/manifest.json"):
            take = json.loads(path.read_text())
            if take["state"] in ("recording", "stopping"):
                take.update(
                    state="interrupted",
                    error="Camera node restarted; stop/finalize this take again",
                )
                save(path, take)
            self.takes[take["id"]] = take
        self.monitor = threading.Thread(target=self.watch, daemon=True)
        self.monitor.start()

    def update(self, take, **values):
        take.update(values)
        save(self.root / take["id"] / "manifest.json", take)

    def status(self):
        with self.lock:
            return dict(
                camera=self.args.name,
                mode=asdict(self.mode),
                audio=self.args.audio,
                active=self.active,
                active_state=self.takes[self.active]["state"] if self.active else None,
                free_bytes=shutil.disk_usage(self.root).free,
                synthetic=bool(self.args.test_video),
                recordings_count=len(self.takes),
            )

    def start(self, body):
        identifier = checked_id(body.get("id"))
        name = checked_name(body.get("name"))
        kind = body.get("kind")
        if kind not in ("calibration", "record"):
            raise ValueError("kind must be calibration or record")
        with self.lock:
            if self.closed.is_set():
                raise ValueError("Camera node is shutting down")
            if identifier in self.takes:
                take = self.takes[identifier]
                if (take["name"], take["kind"]) != (name, kind):
                    raise ValueError(
                        "Recording identity already has different metadata"
                    )
                return dict(take)
            if self.active:
                raise ValueError("Another recording is active")
            if shutil.disk_usage(self.root).free < self.args.min_free_mb * 1024**2:
                raise ValueError("Not enough free disk space")
            if self.capture:
                self.capture.finish()
                self.capture = None
            directory = self.root / identifier
            directory.mkdir()
            take = dict(
                id=identifier,
                name=name,
                kind=kind,
                camera=self.args.name,
                state="recording",
                created=time.time(),
                mode=asdict(self.mode),
                synthetic=bool(self.args.test_video),
            )
            self.takes[identifier] = take
            self.active = identifier
            self.update(take)
            try:
                self.capture = Capture(
                    self.args, self.mode, directory / "video.partial.mkv"
                )
                self.capture.ready()
            except Exception as exc:
                if self.capture:
                    self.capture.finish()
                    self.capture = None
                self.active = None
                self.update(take, state="failed", error=str(exc))
                raise
            return dict(take)

    def stop(self, identifier):
        checked_id(identifier)
        with self.lock:
            take = self.takes[identifier]
            if take["state"] in ("ready", "stopping"):
                return dict(take)
            if self.active and self.active != identifier:
                raise ValueError("Another recording is active")
            if take["state"] not in ("recording", "interrupted"):
                raise ValueError(take.get("error", "Recording failed"))
            if not self.active and self.capture:
                self.capture.finish()
                self.capture = None
            capture = self.capture if self.active == identifier else None
            self.capture = None
            self.active = identifier
            self.update(take, state="stopping")
            self.finalizer = threading.Thread(
                target=self.finalize, args=(take, capture), daemon=True
            )
            self.finalizer.start()
            return dict(take)

    def finalize(self, take, capture):
        try:
            if capture:
                capture.finish()
                if capture.error:
                    raise RuntimeError(capture.error)
            directory = self.root / take["id"]
            partial = directory / "video.partial.mkv"
            video = directory / "video.mkv"
            path = partial if partial.exists() else video
            info = probe(path, self.args.ffprobe)
            digest = sha256(path)
            size = path.stat().st_size
            if path == partial:
                partial.replace(video)
            with self.lock:
                self.update(
                    take,
                    state="ready",
                    error=None,
                    bytes=size,
                    sha256=digest,
                    media=info,
                )
        except Exception as exc:
            with self.lock:
                self.update(take, state="failed", error=str(exc))
        finally:
            with self.lock:
                self.active = None

    def preview(self):
        self.preview_requested = time.monotonic()
        with self.preview_lock:
            now = time.monotonic()
            if self.thumbnail and now - self.thumbnail_time < 1 / self.args.preview_fps:
                return self.thumbnail
            with self.lock:
                if self.active and self.takes[self.active]["state"] != "recording":
                    return None
                if self.capture is None:
                    self.capture = Capture(self.args, self.idle_mode)
                capture = self.capture
                jpeg = capture.jpeg if now - capture.updated < 3 else None
            if not jpeg:
                return None
            # Decode ONE sampled JPEG, never all high-resolution camera frames.
            result = subprocess.run(
                [
                    self.args.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-threads",
                    "1",
                    "-f",
                    "image2pipe",
                    "-i",
                    "pipe:0",
                    "-vf",
                    f"scale=w='min(iw,{self.args.preview_width})':h=-2",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "mjpeg",
                    "-threads",
                    "1",
                    "-q:v",
                    "10",
                    "-f",
                    "image2pipe",
                    "pipe:1",
                ],
                input=jpeg,
                capture_output=True,
                timeout=5,
                check=True,
            )
            self.thumbnail = result.stdout
            self.thumbnail_time = time.monotonic()
            return self.thumbnail

    def watch(self):
        while not self.closed.wait(0.5):
            with self.lock:
                capture = self.capture
                if not capture:
                    continue
                if self.active:
                    if (
                        capture.error
                        or time.monotonic() - capture.progress > 15
                        or shutil.disk_usage(self.root).free
                        < self.args.min_free_mb * 1024**2
                    ):
                        capture.error = (
                            capture.error
                            or "Recording stopped: disk reserve or camera timeout"
                        )
                        self.stop(self.active)
                elif time.monotonic() - self.preview_requested > 5 or capture.error:
                    capture.finish()
                    self.capture = None

    def close(self):
        self.closed.set()
        self.monitor.join(timeout=2)
        with self.lock:
            if self.active and self.takes[self.active]["state"] in (
                "recording",
                "interrupted",
            ):
                self.stop(self.active)
            elif self.capture:
                self.capture.finish()
                self.capture = None
        if self.finalizer:
            self.finalizer.join(timeout=60)


def handler(node):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def json(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return not node.args.token or hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + node.args.token
            )

        def do_GET(self):
            if not self.authorized():
                self.json({"error": "Unauthorized"}, 401)
                return
            path = urlsplit(self.path).path
            try:
                if path == "/api/status":
                    self.json(node.status())
                elif path == "/preview.jpg":
                    jpeg = node.preview()
                    if not jpeg:
                        self.json({"error": "Camera preview is warming up"}, 503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                elif path.startswith("/api/recordings/"):
                    identifier = checked_id(path.rsplit("/", 1)[-1])
                    with node.lock:
                        value = copy.deepcopy(node.takes[identifier])
                    self.json(value)
                elif path.startswith("/files/"):
                    identifier = checked_id(path.rsplit("/", 1)[-1])
                    take = node.takes[identifier]
                    if take["state"] != "ready":
                        raise ValueError("File is not finalized")
                    source = node.root / identifier / "video.mkv"
                    size = source.stat().st_size
                    start = 0
                    if self.headers.get("Range"):
                        match = re.fullmatch(r"bytes=(\d+)-", self.headers["Range"])
                        if not match or int(match[1]) >= size:
                            self.json({"error": "Invalid range"}, 416)
                            return
                        start = int(match[1])
                    self.send_response(206 if self.headers.get("Range") else 200)
                    self.send_header("Content-Type", "video/x-matroska")
                    self.send_header("Content-Length", str(size - start))
                    self.send_header("ETag", '"' + take["sha256"] + '"')
                    self.send_header("Accept-Ranges", "bytes")
                    if self.headers.get("Range"):
                        self.send_header(
                            "Content-Range", f"bytes {start}-{size - 1}/{size}"
                        )
                    self.end_headers()
                    with source.open("rb") as stream:
                        stream.seek(start)
                        while block := stream.read(256 * 1024):
                            self.wfile.write(block)
                else:
                    self.json({"error": "Unknown route"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except KeyError:
                self.json({"error": "Unknown recording"}, 404)
            except Exception as exc:
                self.json({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.authorized():
                self.json({"error": "Unauthorized"}, 401)
                return
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
                    self.json(node.start(body))
                elif path == "/api/stop":
                    self.json(node.stop(body.get("id")))
                else:
                    self.json({"error": "Unknown route"}, 404)
            except KeyError:
                self.json({"error": "Unknown recording"}, 404)
            except Exception as exc:
                self.json({"error": str(exc)}, 400)

    return Handler


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="cam1")
    parser.add_argument("--device")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--resolution", default="max", help="max (default) or WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--fps", type=rate, help="Default: maximum FPS at the selected resolution"
    )
    parser.add_argument(
        "--input-format", choices=["auto", "mjpeg", "h264"], default="auto"
    )
    parser.add_argument(
        "--audio", default="default", help="ALSA input, e.g. plughw:CARD=Camera,DEV=0"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-formats", action="store_true")
    parser.add_argument("--list-audio", action="store_true")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(__file__).resolve().parent / "runtime" / "recordings",
    )
    parser.add_argument(
        "--ffmpeg",
        default=shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg"),
    )
    parser.add_argument(
        "--ffprobe",
        default=shutil.which("ffprobe") or str(Path(sys.executable).parent / "ffprobe"),
    )
    parser.add_argument("--v4l2-ctl", default="v4l2-ctl")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--preview-width", type=int, default=320)
    parser.add_argument("--preview-fps", type=float, default=1)
    parser.add_argument("--min-free-mb", type=int, default=256)
    parser.add_argument(
        "--test-video",
        type=Path,
        help="Diagnostic prerecorded MJPEG+audio file, instead of physical hardware",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
        parser.error("Invalid camera name")
    if (
        not 1 <= args.port <= 65535
        or not 64 <= args.preview_width <= 640
        or not 0 < args.preview_fps <= 2
        or args.min_free_mb < 16
    ):
        parser.error("Invalid port/preview/disk settings")
    if args.resolution != "max":
        args.resolution = resolution(args.resolution)
    args.token = args.token_file.read_text().strip() if args.token_file else ""
    return args


def main():
    args = arguments()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.list_audio:
        return subprocess.call(["arecord", "-L"])
    if args.list_devices:
        for device, name, modes, _ in discover_devices(args.v4l2_ctl):
            print(
                device,
                name,
                sorted({(m.width, m.height, m.fps, m.pixel_format) for m in modes}),
            )
        return 0
    if args.test_video:
        info = probe(args.test_video, args.ffprobe)
        if info["video_codec"] != "mjpeg":
            raise ValueError("--test-video requires MJPEG video and audio")
        mode = idle = Mode("mjpeg", info["width"], info["height"], info["fps"])
    else:
        args.device, modes, text = choose_device(args, args.v4l2_ctl)
        if args.list_formats:
            print(text)
            return 0
        features = subprocess.run(
            [args.ffmpeg, "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if "-isync" not in features:
            raise ValueError(
                "FFmpeg with -isync is required on cameras (FFmpeg 5.1+); install a newer build"
            )
        mode = select_mode(modes, args.resolution, args.fps, args.input_format)
        idle = min(
            (m for m in modes if m.pixel_format == "mjpeg"),
            key=lambda m: (abs(m.width * m.height - 320 * 240), m.fps),
            default=mode,
        )
    node = Node(args, mode, idle)
    http = ThreadingHTTPServer((args.host, args.port), handler(node))
    http.daemon_threads = True
    LOG.info(
        "Camera %s: %sx%s @ %s, %s copy + PCM audio. API on port %s",
        args.name,
        mode.width,
        mode.height,
        mode.fps,
        mode.pixel_format,
        args.port,
    )

    def stop(*_):
        threading.Thread(target=http.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        http.serve_forever()
    finally:
        http.server_close()
        node.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc))
