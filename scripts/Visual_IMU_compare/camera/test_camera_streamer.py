"""Hardware-free checks; FFmpeg tests use a deterministic audio/video fixture."""

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
import wave

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "camera_node_tested", HERE / "camera_streamer.py"
)
cam = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cam
spec.loader.exec_module(cam)


def wait_for(fn, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for operation")


def request(base, path, body=None, headers=None):
    options = dict(headers or {})
    if body is not None:
        options["Content-Type"] = "application/json"
    return urlopen(
        Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=options,
        ),
        timeout=30,
    )


def make_video(directory, silent=False):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg required for recording integration tests")
    directory.mkdir(parents=True, exist_ok=True)
    wav = directory / "sound.wav"
    rng = random.Random(105)
    # Aperiodic common sound: unlike a sine tone, this has one unambiguous lag.
    with wave.open(str(wav), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(
            b"".join(
                struct.pack("<h", 0 if silent else rng.randint(-10000, 10000))
                for _ in range(16000 * 8)
            )
        )
    target = directory / "source.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=15:duration=8",
            "-i",
            str(wav),
            "-c:v",
            "mjpeg",
            "-threads",
            "1",
            "-q:v",
            "3",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            "-y",
            str(target),
        ],
        check=True,
    )
    return target


@contextmanager
def node_server(tmp_path, source, name="cam1", token=""):
    args = cam.arguments(
        ["--name", name, "--records", str(tmp_path / name), "--test-video", str(source)]
    )
    args.token = token
    mode = cam.Mode("mjpeg", 320, 240, 15)
    node = cam.Node(args, mode, mode)
    http = ThreadingHTTPServer(("127.0.0.1", 0), cam.handler(node))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield node, f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        node.close()
        thread.join(5)


def test_modes_max_and_exact_fps():
    modes = [
        cam.Mode("mjpeg", 640, 480, 30),
        cam.Mode("mjpeg", 1920, 1080, 15),
        cam.Mode("yuyv422", 4096, 2160, 5),
    ]
    assert cam.select_mode(modes).width == 1920
    assert cam.select_mode(modes, fps=30).width == 640
    with pytest.raises(ValueError):
        cam.select_mode(modes, (1920, 1080), 30)
    text = "[0]: 'MJPG'\n Size: Discrete 1920x1080\n Interval: Discrete 0.067s (15.000 fps)"
    assert cam.parse_modes(text) == [modes[1]]


@pytest.mark.parametrize(
    "bad", ["../foo", "/tmp/file", "name/child", ".", None, "x" * 81]
)
def test_names_reject_paths(bad):
    with pytest.raises(ValueError):
        cam.checked_name(bad)


def test_command_copies_camera_and_keeps_av_timing():
    args = cam.arguments(
        ["--device", "/dev/video0", "--audio", "plughw:CARD=Camera,DEV=0"]
    )
    cmd = cam.command(args, cam.Mode("mjpeg", 1920, 1080, 30), Path("/tmp/video.mkv"))
    assert cmd.count("copy") == 2
    assert "-isync" in cmd and cmd[cmd.index("-isync") + 1] == "0"
    assert "-timestamps" in cmd and "abs" in cmd
    assert "-copyts" in cmd and "-start_at_zero" in cmd
    assert "aresample=async=1:first_pts=0" in cmd
    assert "-vf" not in cmd  # No full-resolution decode/scale at the camera FPS.


def test_real_recording_preview_and_range_transfer(tmp_path):
    source = make_video(tmp_path / "fixture")
    with node_server(tmp_path, source, token="secret") as (node, base):
        with pytest.raises(HTTPError) as error:
            request(base, "/api/status")
        assert error.value.code == 401
        headers = {"Authorization": "Bearer secret"}
        body = dict(id=uuid.uuid4().hex, name="Новая запись 01", kind="record")
        with request(base, "/api/start", body, headers) as response:
            assert json.load(response)["state"] == "recording"
        with request(base, "/api/start", body, headers) as response:
            assert json.load(response)["id"] == body["id"]
        with pytest.raises(HTTPError):
            request(base, "/files/" + body["id"], headers=headers)
        with request(base, "/preview.jpg", headers=headers) as response:
            preview = response.read()
            assert preview.startswith(b"\xff\xd8") and len(preview) < 40000
        time.sleep(0.8)
        with request(base, "/api/stop", {"id": body["id"]}, headers) as response:
            assert json.load(response)["state"] == "stopping"
        take = wait_for(
            lambda: (
                node.takes[body["id"]]
                if node.takes[body["id"]]["state"] in ("ready", "failed")
                else None
            )
        )
        assert take["state"] == "ready", take
        assert take["media"]["audio_codec"] == "pcm_s16le"
        assert take["media"]["width"] == 320
        video = node.root / body["id"] / "video.mkv"
        data = video.read_bytes()
        assert cam.sha256(video) == take["sha256"]
        with request(
            base, "/files/" + body["id"], headers={**headers, "Range": "bytes=101-"}
        ) as response:
            assert response.status == 206
            assert response.read() == data[101:]
        with request(
            base, "/files/" + body["id"], headers={**headers, "Range": "bytes=0-"}
        ) as response:
            assert response.status == 206
            assert response.headers["Content-Range"].startswith("bytes 0-")
            assert response.read() == data
        with pytest.raises(HTTPError) as error:
            request(
                base,
                "/files/" + body["id"],
                headers={**headers, "Range": f"bytes={len(data)}-"},
            )
        assert error.value.code == 416
        assert node.capture is None


def test_recovery_finalizes_interrupted_file(tmp_path):
    source = make_video(tmp_path / "fixture")
    identifier = uuid.uuid4().hex
    root = tmp_path / "cam1"
    folder = root / identifier
    folder.mkdir(parents=True)
    shutil.copyfile(source, folder / "video.partial.mkv")
    cam.save(
        folder / "manifest.json",
        dict(
            id=identifier,
            name="recovery",
            kind="record",
            camera="cam1",
            state="recording",
            created=time.time(),
        ),
    )
    with node_server(tmp_path, source) as (node, base):
        assert node.takes[identifier]["state"] == "interrupted"
        with request(base, "/api/stop", {"id": identifier}):
            pass
        wait_for(lambda: node.takes[identifier]["state"] != "stopping")
        assert node.takes[identifier]["state"] == "ready"
        assert (folder / "video.mkv").exists()
