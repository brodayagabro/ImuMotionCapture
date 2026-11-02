"""Server-side RTSP recording. Preserve H.264 packets and their media timestamps."""
from __future__ import annotations

import asyncio
import csv
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


@lru_cache(maxsize=8)
def socket_timeout_option(binary):
    """FFmpeg 4 uses stimeout; its timeout option enables RTSP LISTEN mode."""
    result = subprocess.run([binary, "-hide_banner", "-h", "demuxer=rtsp"],
                            capture_output=True, text=True, timeout=5, check=True)
    help_text = result.stdout + result.stderr
    if re.search(r"^\s+-stimeout\s", help_text, re.MULTILINE):
        return "-stimeout"
    if re.search(r"^\s+-timeout\s+.*socket.*I/O", help_text, re.MULTILINE):
        return "-timeout"
    raise ValueError(f"Cannot determine RTSP socket timeout option for {binary}")


async def input_options(binary, timeout):
    option = await asyncio.to_thread(socket_timeout_option, binary)
    return ["-rtsp_transport", "tcp", option, str(int(timeout * 1e6)),
            "-analyzeduration", "1000000", "-probesize", "1000000"]


async def terminate(process):
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


async def probe(source, ffprobe="ffprobe", timeout=10, *, recorded=False):
    command = [ffprobe, "-v", "error"]
    if not recorded:
        command += await input_options(ffprobe, timeout)
    else:
        command += ["-count_packets"]
    command += ["-select_streams", "v:0", "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration",
                "-of", "json", str(source)]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout + 2)
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace")[-1200:] or "ffprobe failed")
        data = json.loads(stdout)
        video = data["streams"][0]
        if video["codec_name"] != "h264":
            raise ValueError("RTSP recording requires H.264; use camera_streamer.py")
        fps = 0.0
        for key in ("avg_frame_rate", "r_frame_rate"):
            try:
                fps = float(Fraction(video.get(key, "0/1")))
            except (ValueError, ZeroDivisionError):
                continue
            if fps > 0:
                break
        if not fps > 0 or not video.get("width") or not video.get("height"):
            raise ValueError("No usable video size / FPS in RTSP stream")
        result = dict(resolution=f"{video['width']}x{video['height']}", fps=fps, codec="h264")
        try:
            nominal = float(Fraction(video.get("r_frame_rate", "0/1")))
        except (ValueError, ZeroDivisionError):
            nominal = 0
        result["nominal_fps"] = nominal if 0 < nominal <= 240 else fps
        if recorded:
            result.update(packets=int(video.get("nb_read_packets", 0)),
                          duration=float(data.get("format", {}).get("duration", 0)))
            if result["packets"] < 10 or result["duration"] <= 0:
                raise ValueError("Recording is empty or too short (at least 10 frames required)")
        return result
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("No readable H.264 video") from exc
    finally:
        await terminate(process)


class StreamRecording:
    def __init__(self, name, url, path, ffmpeg, timeout):
        self.name, self.url, self.path = name, url, path
        self.ffmpeg, self.timeout = ffmpeg, timeout
        self.process = None
        self.reader = None
        self.timing_reader = None
        self.timing_ready = asyncio.Event()
        self.ready = asyncio.Event()
        self.media_time = 0.0
        self.last_frame = time.monotonic()
        self.stopping = False
        self.error = None

    async def start(self):
        # Keep the original H.264. A second output decodes the SAME input and
        # logs each frame, so the host can timestamp it without a second RTSP connection.
        input_args = await input_options(self.ffmpeg, self.timeout)
        passthrough = ["-vsync", "0"] if "-stimeout" in input_args else ["-fps_mode", "passthrough"]
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "info", "-nostats",
                   "-stats_period", "0.2", "-progress", "pipe:1",
                   "-threads", "1", *input_args,
                   "-i", self.url, "-map", "0:v:0", "-an", "-c:v", "copy",
                   "-avoid_negative_ts", "make_zero", "-movflags",
                   "+frag_keyframe+empty_moov+default_base_moof", "-n", str(self.path),
                   "-map", "0:v:0", "-an", "-vf", "showinfo=checksum=0", "-threads:v", "1", *passthrough,
                   "-f", "null", "/dev/null"]
        self.process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.reader = asyncio.create_task(self.read_progress())
        self.timing_reader = asyncio.create_task(self.read_timing())
        await asyncio.wait_for(asyncio.gather(self.ready.wait(), self.timing_ready.wait()), self.timeout + 5)
        if self.error or self.media_time <= 0:
            raise RuntimeError(self.error or f"{self.name}: no video received")

    async def read_timing(self):
        pattern = re.compile(rb"\[Parsed_showinfo_[^\]]+\].*\bn:\s*(\d+).*\bpts_time:([^\s]+)")
        try:
            with self.path.with_suffix(".log").open("wb") as log, self.path.with_suffix(".timestamps.csv").open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_index", "receive_monotonic_s", "source_pts_s"])
                while line := await self.process.stderr.readline():
                    received = time.monotonic()
                    match = pattern.search(line)
                    if match:
                        writer.writerow([int(match[1]), f"{received:.9f}", match[2].decode()])
                        stream.flush()  # Preserve timing alongside fragmented MP4 on interruption.
                        self.timing_ready.set()
                    elif b"Parsed_showinfo_" not in line:
                        log.write(line)
        except Exception as exc:
            self.error = f"{self.name}: cannot capture frame timestamps: {exc}"
            self.timing_ready.set()
            # Drain the pipe while the supervisor stops FFmpeg; otherwise a
            # full stderr pipe can prevent MP4 finalization after a disk error.
            while await self.process.stderr.read(65536):
                pass
        finally:
            self.timing_ready.set()

    async def read_progress(self):
        while line := await self.process.stdout.readline():
            key, _, value = line.decode(errors="replace").strip().partition("=")
            # FFmpeg 6 does not emit `frame` in progress for stream copy.
            if key == "out_time_us":
                try:
                    media_time = int(value) / 1e6
                except ValueError:
                    continue
                if media_time > self.media_time:
                    self.media_time, self.last_frame = media_time, time.monotonic()
                    self.ready.set()
        await self.process.wait()
        if not self.stopping:
            self.error = f"{self.name}: RTSP stream ended (FFmpeg {self.process.returncode}); see capture log"
        self.ready.set()

    def failure(self):
        if self.error:
            return self.error
        if self.process and self.process.returncode is not None and not self.stopping:
            return f"{self.name}: RTSP stream ended (FFmpeg {self.process.returncode})"
        if not self.stopping and time.monotonic() - self.last_frame > self.timeout:
            return f"{self.name}: no new RTSP frames for {self.timeout:g} seconds"
        return None

    async def stop(self):
        self.error = self.failure()
        self.stopping = True
        if self.process is None:
            return
        if self.process.returncode is None:
            try:
                self.process.stdin.write(b"q\n")
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), self.timeout + 2)
            except (BrokenPipeError, ConnectionResetError):
                await self.process.wait()
            except asyncio.TimeoutError:
                self.error = f"{self.name}: FFmpeg did not finalize the recording"
                await terminate(self.process)
        if self.reader:
            await self.reader
        if self.timing_reader:
            await self.timing_reader
        if self.process.returncode != 0 and not self.error:
            self.error = f"{self.name}: FFmpeg exited {self.process.returncode}; recording incomplete"


class RTSPRecorder:
    def __init__(self, cameras, folder, ffmpeg="ffmpeg", ffprobe="ffprobe", timeout=10):
        self.folder, self.ffprobe, self.timeout = folder, ffprobe, timeout
        destination = folder / "rtsp_capture"
        destination.mkdir(exist_ok=True)
        self.streams = {name: StreamRecording(name, url, destination / f"{name}.mp4", ffmpeg, timeout)
                        for name, url in cameras.items()}

    async def start(self):
        results = await asyncio.gather(*(s.start() for s in self.streams.values()), return_exceptions=True)
        errors = [f"{name}: {str(value) or type(value).__name__}" for name, value in zip(self.streams, results)
                  if isinstance(value, BaseException)]
        if errors:
            await self.stop()
            raise RuntimeError("; ".join(errors))
        (self.folder / "capture_timing.json").write_text(json.dumps(dict(
            time_basis="server_monotonic_after_decode_log_received",
            recording_start_monotonic_s=time.monotonic(),
            hardware_capture_timestamps=False)))

    def failure(self):
        return next((error for s in self.streams.values() if (error := s.failure())), None)

    async def stop(self):
        await asyncio.gather(*(s.stop() for s in self.streams.values()))

    async def finalize(self, camera_names, take_id):
        error = self.failure()
        if error:
            raise RuntimeError(error)
        return await finalize_recordings(self.folder, camera_names, take_id, self.ffprobe)


async def finalize_recordings(folder: Path, camera_names, take_id, ffprobe="ffprobe"):
    """Also used after a host restart; fragmented MP4s preserve complete fragments."""
    paths = {name: folder / "rtsp_capture" / f"{name}.mp4" for name in camera_names}
    results = await asyncio.gather(*(probe(path, ffprobe, timeout=60, recorded=True) for path in paths.values()))
    raw = folder / "raw_videos"
    raw.mkdir(exist_ok=True)
    for name, source in paths.items():
        target = raw / f"{take_id}.id-{name}.idx-0.mp4"
        if not target.exists():
            partial = target.with_suffix(".partial")
            await asyncio.to_thread(shutil.copyfile, source, partial)
            partial.replace(target)
    camera_info = dict(zip(camera_names, results))
    for name, info in camera_info.items():
        info.update(video_name=f"{take_id}.id-{name}.idx-0",
                    timing_file=f"rtsp_capture/{name}.timestamps.csv")
    report = dict(transport="rtsp", encoding="copy", hardware_capture_timestamps=False,
                  timestamps="RTSP media PTS; not camera exposure or synchronized host clocks",
                  cameras=camera_info)
    (folder / "capture_report.json").write_text(json.dumps(report, indent=2))
    return report
