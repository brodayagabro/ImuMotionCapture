"""Real RTSP publishers → MediaMTX → host MP4 → native flash synchronization."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

import freemocap_host as host_module
from rtsp_recording import RTSPRecorder
from rtsp_relay import relay_config

HERE = Path(__file__).resolve().parent


@pytest.fixture
def rtsp_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(HERE / ".tools/bin") + os.pathsep + os.environ["PATH"])
    ffmpeg = shutil.which("ffmpeg")
    mediamtx = shutil.which("mediamtx")
    if not ffmpeg or not mediamtx or not shutil.which("ffprobe"):
        pytest.skip("Real RTSP tests require FFmpeg, ffprobe and MediaMTX")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "mediamtx.yml"
    config.write_text(relay_config("127.0.0.1", port))
    cameras = {name: f"rtsp://127.0.0.1:{port}/{name}" for name in ("cam1", "cam2")}
    stop, flash = threading.Event(), threading.Event()
    publisher = thread = None
    with (tmp_path / "relay.log").open("wb") as relay_log, (tmp_path / "publisher.log").open("wb") as pub_log:
        relay = subprocess.Popen([mediamtx, str(config)], stdout=relay_log, stderr=relay_log)
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.05)
            else:
                pytest.fail("MediaMTX did not start")
            # One physical event visible in both independent RTSP paths.
            publisher = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", "160x120", "-framerate", "15", "-i", "pipe:0", "-map", "0:v:0",
                "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p", "-g", "15", "-bf", "0", "-f", "tee",
                "|".join(f"[f=rtsp:rtsp_transport=tcp]{url}" for url in cameras.values())],
                stdin=subprocess.PIPE, stdout=pub_log, stderr=pub_log)

            def frames():
                while not stop.is_set():
                    try:
                        publisher.stdin.write(bytes([235 if flash.is_set() else 25]) * (160 * 120 * 3))
                        publisher.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        return
                    stop.wait(1 / 15)
            thread = threading.Thread(target=frames, daemon=True)
            thread.start()
            time.sleep(.4)
            yield SimpleNamespace(cameras=cameras, flash=flash, publisher=publisher)
        finally:
            stop.set()
            for process in (publisher, relay):
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if thread:
                thread.join(timeout=2)


def arguments(tmp_path, sources):
    return SimpleNamespace(camera=[f"{n}={u}" for n, u in sources.cameras.items()],
                           records=tmp_path / "host", output=tmp_path / "output", blender="blender",
                           capture_timeout=3, ffmpeg=shutil.which("ffmpeg"), ffprobe=shutil.which("ffprobe"))


async def show_flash(sources):
    await asyncio.sleep(.7)
    sources.flash.set()
    await asyncio.sleep(.4)
    sources.flash.clear()
    await asyncio.sleep(1.0)


def test_rtsp_urls_and_mixed_protocol_rejection():
    assert host_module.camera_endpoints(["a=rtsp://localhost:8554/a", "b=rtsp://localhost:8554/b"])
    for values in [["a=rtsp://x/a", "b=http://x"], ["a=rtsp://x/a", "b=rtsp://x/a"],
                   ["a=rtsp://user:password@x/a", "b=rtsp://x/b"]]:
        with pytest.raises(ValueError):
            host_module.camera_endpoints(values)


def test_rtsp_recording_to_native_sync_and_recovery(tmp_path, rtsp_sources):
    async def run():
        args = arguments(tmp_path, rtsp_sources)
        host = host_module.Host(args)
        try:
            take = await host.start(host_module.StartRequest(kind="sync"))
            assert take["state"] == "recording"
            assert all(info["received_seconds"] > 0 for info in await host.health())
            await show_flash(rtsp_sources)
            await host.stop(take["id"])
            await asyncio.wait_for(host.task, 60)
            assert take["state"] == "complete", take
            folder = host.root / take["id"]
            report = json.loads((folder / "capture_report.json").read_text())
            assert report["encoding"] == "copy" and not report["hardware_capture_timestamps"]
            assert all(info["packets"] > 10 for info in report["cameras"].values())
            aligned = json.loads((folder / "sync_result.json").read_text())["aligned_flash_frames"]
            assert max(aligned) - min(aligned) <= 1
            assert len(list((folder / "raw_videos").glob("*.mp4"))) == 2

            # Graceful shutdown finalizes files. A restarted host can finish processing.
            take = await host.start(host_module.StartRequest(kind="sync"))
            await show_flash(rtsp_sources)
            await host.close()
            host = host_module.Host(args)
            take = host.get(take["id"])
            assert take["state"] == "interrupted"
            await host.stop(take["id"])
            await asyncio.wait_for(host.task, 60)
            assert take["state"] == "complete", take
        finally:
            await host.close()
    asyncio.run(run())


def test_stream_loss_stops_all_receivers_and_rejects_processing(tmp_path, rtsp_sources):
    async def run():
        host = host_module.Host(arguments(tmp_path, rtsp_sources))
        try:
            take = await host.start(host_module.StartRequest(kind="sync"))
            rtsp_sources.publisher.terminate()
            for _ in range(100):
                if take["state"] == "failed":
                    break
                await asyncio.sleep(.1)
            assert take["state"] == "failed", take
            assert take["capture_failed"]
            assert all(s.process.returncode is not None for s in host.capture.streams.values())
            with pytest.raises(host_module.HTTPException, match="RTSP recording failed"):
                await host.stop(take["id"])
            assert not (host.root / take["id"] / "raw_videos").exists()
        finally:
            await host.close()
    asyncio.run(run())


def test_partial_rtsp_start_failure_cleans_up_every_process(tmp_path, rtsp_sources):
    async def run():
        urls = dict(rtsp_sources.cameras)
        urls["cam2"] += "-absent"
        recorder = RTSPRecorder(urls, tmp_path, shutil.which("ffmpeg"), shutil.which("ffprobe"), timeout=3)
        try:
            with pytest.raises(RuntimeError):
                await recorder.start()
            assert all(s.process.returncode is not None for s in recorder.streams.values())
        finally:
            await recorder.stop()
    asyncio.run(run())
