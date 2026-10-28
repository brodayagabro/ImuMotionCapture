#!/usr/bin/env python3
"""Send a Linux V4L2 camera to a server over RTSP/TCP.

Camera computer:
  python3 camera_streamer.py --server 192.168.1.10 --name cam1 \
      --device /dev/video0 --resolution 1280x720 --fps 30

GPU server (RTSP relay, separate terminal/service):
  python3 camera_streamer.py --relay --mediamtx /path/to/mediamtx

The existing visual_mocap_server.py then reads rtsp://127.0.0.1:8554/cam1.
Python >= 3.9; camera side needs ffmpeg and v4l2-ctl, no Python packages.
See CAMERA_STREAMER.md for installation, supported modes and deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time

LOG = logging.getLogger("camera-streamer")
PIXEL_FORMATS = {"MJPG": "mjpeg", "YUYV": "yuyv422", "H264": "h264", "NV12": "nv12"}


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
        raise argparse.ArgumentTypeError("H.264 dimensions must be even, between 16 and 8192")
    return width, height


def rate(value):
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError("FPS must be a number or fraction such as 30000/1001") from None
    if not 0 < fps <= 240:
        raise argparse.ArgumentTypeError("FPS must be > 0 and <= 240")
    return fps


def host(value):
    value = value.strip("[]")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        if len(value) > 253 or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                                      for part in value.split(".")):
            raise argparse.ArgumentTypeError("Use an IP address or hostname, without rtsp:// or port") from None
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", type=host, help="Destination GPU server IP/hostname")
    parser.add_argument("--port", type=int, default=8554, help="RTSP port on the server")
    parser.add_argument("--name", default="cam1", help="Unique stream name on the server")
    parser.add_argument("--device", "--camera", help="V4L2 path or index; auto-detected if exactly one camera is present")
    parser.add_argument("--resolution", "--size", type=resolution, help="Camera resolution, e.g. 1280x720")
    parser.add_argument("--fps", type=rate, help="Camera frame rate, e.g. 30 or 30000/1001")
    parser.add_argument("--input-format", choices=["auto", *PIXEL_FORMATS.values()], default="auto")
    parser.add_argument("--encoder", choices=["libx264", "h264_v4l2m2m", "copy"], default="libx264")
    parser.add_argument("--bitrate", default="4M", help="H.264 target/max bitrate, e.g. 4M or 1500k")
    parser.add_argument("--ffmpeg", help="FFmpeg executable; defaults to PATH, then optional imageio-ffmpeg")
    parser.add_argument("--v4l2-ctl", default="v4l2-ctl", help="Path to v4l2-ctl")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-formats", action="store_true", help="Show supported modes of the selected camera")
    parser.add_argument("--retry-delay", type=float, default=2)
    parser.add_argument("--stall-timeout", type=float, default=15, help="Restart if no new encoded frames for this many seconds")
    parser.add_argument("--no-reconnect", action="store_true", help="Exit on a camera/network error")
    parser.add_argument("--duration", type=float, help="Stop after N wall-clock seconds; useful for testing")
    parser.add_argument("--relay", action="store_true", help="Run only the MediaMTX relay on the GPU server")
    parser.add_argument("--bind", type=host, default="0.0.0.0", help="Relay listening address")
    parser.add_argument("--mediamtx", default="mediamtx", help="MediaMTX executable for --relay")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
        parser.error("--name must contain 1–48 letters, digits, '_' or '-'")
    if not re.fullmatch(r"[1-9]\d*(?:[kKmM])?", args.bitrate):
        parser.error("--bitrate must be a positive integer optionally followed by k or M")
    for field in ("retry_delay", "stall_timeout", "duration"):
        number = getattr(args, field)
        if number is not None and (not math.isfinite(number) or number <= 0):
            parser.error(f"--{field.replace('_', '-')} must be finite and positive")
    if args.relay and (args.server or args.device or args.resolution or args.fps or args.list_devices or args.list_formats):
        parser.error("--relay runs on the server and cannot be combined with camera capture options")
    if not (args.relay or args.list_devices or args.list_formats):
        if not args.server or args.resolution is None or args.fps is None:
            parser.error("Camera streaming requires --server, --resolution and --fps")
    if args.encoder == "copy" and args.input_format not in ("auto", "h264"):
        parser.error("--encoder copy requires an H.264-capable camera (--input-format h264)")
    return args


def executable(value):
    found = shutil.which(str(value))
    if not found:
        raise ValueError(f"Executable not found: {value}")
    return str(Path(found).resolve())


def find_ffmpeg(requested=None):
    if requested:
        return executable(requested)
    if shutil.which("ffmpeg"):
        return executable("ffmpeg")
    try:
        import imageio_ffmpeg
        return executable(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        raise ValueError("FFmpeg not found. Ubuntu/Debian: sudo apt install ffmpeg v4l-utils") from None


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
    result = subprocess.run([tool, "--device", str(device), "--list-formats-ext"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise ValueError(f"Cannot query {device}: {result.stderr.strip() or result.stdout.strip()}")
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
        path = Path("/dev/video" + args.device) if args.device.isdecimal() else Path(args.device)
        modes, text = probe_device(tool, path)
        return path, modes, text
    cameras = discover_devices(tool)
    if len(cameras) != 1:
        available = ", ".join(str(c[0]) for c in cameras) or "none"
        raise ValueError(f"Specify --device; usable cameras: {available}. Check --list-devices and device permissions.")
    path, _, modes, text = cameras[0]
    return path, modes, text


def choose_format(args, modes):
    if args.encoder == "copy":
        preferences = ["h264"]
    elif args.input_format == "auto":
        preferences = ["mjpeg", "yuyv422", "h264", "nv12"]
    else:
        preferences = [args.input_format]
    for pixel_format in preferences:
        for mode in modes:
            if (mode.pixel_format == pixel_format and (mode.width, mode.height) == args.resolution
                    and math.isclose(mode.fps, float(args.fps), rel_tol=1e-4, abs_tol=.001)):
                return pixel_format
    width, height = args.resolution
    raise ValueError(f"Camera does not advertise {width}x{height} @ {float(args.fps):g} fps "
                     f"in {', '.join(preferences)}. Run --list-formats; resolution/FPS will not be silently changed.")


def rtsp_url(server, port, name):
    address = f"[{server}]" if ":" in server else server
    return f"rtsp://{address}:{port}/{name}"


def ffmpeg_command(args, binary, device, pixel_format):
    width, height = args.resolution
    fps = str(args.fps)
    command = [binary, "-hide_banner", "-nostdin", "-loglevel", "warning", "-nostats",
               "-stats_period", "1", "-progress", "pipe:1", "-thread_queue_size", "4",
               "-f", "v4l2", "-input_format", pixel_format, "-video_size", f"{width}x{height}",
               "-framerate", fps, "-i", str(device), "-map", "0:v:0", "-an"]
    if args.encoder == "copy":
        command += ["-c:v", "copy"]
    else:
        command += ["-c:v", args.encoder, "-pix_fmt", "yuv420p", "-b:v", args.bitrate,
                    "-maxrate", args.bitrate, "-bufsize", args.bitrate,
                    "-g", str(max(1, round(float(args.fps)))), "-bf", "0"]
        if args.encoder == "libx264":
            command += ["-preset", "ultrafast", "-tune", "zerolatency", "-sc_threshold", "0"]
        # No CFR resampling: do not duplicate frames to disguise camera drops.
        command += ["-vsync", "0"]
    command += ["-flush_packets", "1", "-f", "rtsp", "-rtsp_transport", "tcp",
                rtsp_url(args.server, args.port, args.name)]
    return command


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def stream_attempt(command, stop, deadline, stall_timeout):
    """Drain both pipes, detect stalls by changing frame count, never buffer video in Python."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    buffers, progress = {}, {}
    frames, last_frame = 0, time.monotonic()
    stalled = False
    try:
        with selectors.DefaultSelector() as selector:
            for pipe, name in ((process.stdout, "progress"), (process.stderr, "stderr")):
                selector.register(pipe, selectors.EVENT_READ, name)
                buffers[name] = b""
            while not stop.is_set() and time.monotonic() < deadline:
                for key, _ in selector.select(timeout=.2):
                    block = os.read(key.fd, 4096)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    buffers[name] += block
                    while b"\n" in buffers[name]:
                        raw, buffers[name] = buffers[name].split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace").strip()
                        if name == "stderr":
                            if line:
                                LOG.warning("FFmpeg: %s", line[:2000])
                        else:
                            field, sep, value = line.partition("=")
                            if sep:
                                progress[field] = value
                            if field == "frame" and value.isdecimal() and int(value) > frames:
                                frames, last_frame = int(value), time.monotonic()
                            if field == "progress":
                                LOG.info("Frames=%s, encoded fps=%s, speed=%s", progress.get("frame", "0"),
                                         progress.get("fps", "?"), progress.get("speed", "?"))
                    if len(buffers[name]) > 16384:
                        buffers[name] = buffers[name][-16384:]
                if process.poll() is not None and not selector.get_map():
                    break
                if time.monotonic() - last_frame > stall_timeout:
                    LOG.error("No new frames for %.1f seconds; restarting capture", stall_timeout)
                    stalled = True
                    break
    finally:
        stop_child(process)
        process.stdout.close()
        process.stderr.close()
    return (1 if stalled else process.returncode), frames


def publish(args, stop):
    tool = executable(args.v4l2_ctl)
    if args.list_devices:
        for path, name, _, _ in discover_devices(tool):
            print(f"{path}: {name}")
        return 0
    device, modes, text = choose_device(args, tool)
    if args.list_formats:
        print(f"{device}\n{text}")
        return 0
    pixel_format = choose_format(args, modes)
    command = ffmpeg_command(args, find_ffmpeg(args.ffmpeg), device, pixel_format)
    destination = rtsp_url(args.server, args.port, args.name)
    LOG.info("%s → %s; %sx%s @ %s fps; input=%s, encoder=%s", device, destination,
             *args.resolution, args.fps, pixel_format, args.encoder)
    deadline = time.monotonic() + args.duration if args.duration else math.inf
    total = 0
    while not stop.is_set() and time.monotonic() < deadline:
        code, frames = stream_attempt(command, stop, deadline, args.stall_timeout)
        total += frames
        if stop.is_set() or time.monotonic() >= deadline:
            break
        if args.no_reconnect:
            return code if code > 0 else 1
        LOG.warning("Stream ended (code %s). Retrying in %.1f seconds", code, args.retry_delay)
        stop.wait(max(0, min(args.retry_delay, deadline-time.monotonic())))
    LOG.info("Stopped. Encoded frames across attempts: %s", total)
    return 0 if total or stop.is_set() else 1


def relay_config(bind, port):
    address = f"[{bind}]:{port}" if ":" in bind else f"{bind}:{port}"
    return ("logLevel: info\nrtsp: yes\nrtspTransports: [tcp]\nrtspAddress: " + json.dumps(address) + "\n"
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\napi: no\nmetrics: no\npprof: no\nplayback: no\n"
            "pathDefaults:\n  source: publisher\n  overridePublisher: false\npaths:\n  all_others:\n")


def relay(args, stop):
    binary = executable(args.mediamtx)
    with tempfile.TemporaryDirectory(prefix="visual-mocap-relay-") as directory:
        config = Path(directory) / "mediamtx.yml"
        config.write_text(relay_config(args.bind, args.port))
        # MediaMTX may create certificates in cwd. Keep them out of the repo.
        process = subprocess.Popen([binary, str(config)], cwd=directory, start_new_session=True)
        try:
            deadline = time.monotonic()+args.duration if args.duration else math.inf
            LOG.info("RTSP relay on %s:%s; streams: rtsp://SERVER:%s/CAMERA_NAME", args.bind, args.port, args.port)
            while process.poll() is None and not stop.is_set() and time.monotonic() < deadline:
                stop.wait(.2)
            if process.poll() is not None:
                return process.returncode
        finally:
            stop_child(process)
    return 0


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    try:
        return relay(args, stop) if args.relay else publish(args, stop)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
