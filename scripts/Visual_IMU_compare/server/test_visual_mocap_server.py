"""Real HTTP transfers and the native SkellySync pipeline; no physical cameras needed."""

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
import uuid

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "camera"))
from test_camera_streamer import make_video, node_server, request, wait_for

spec = importlib.util.spec_from_file_location(
    "recording_host_tested", HERE / "visual_mocap_server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@contextmanager
def host_server(root, bases):
    args = SimpleNamespace(
        camera=[f"{name}={base}" for name, base in bases.items()],
        records=root,
        camera_token="",
    )
    host = server.Host(args)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(host))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        host.close()
        thread.join(5)


def test_camera_configuration_and_names():
    assert (
        len(server.cameras(["a=http://localhost:8081", "b=http://localhost:8082"])) == 2
    )
    for items in (
        ["a=http://localhost:8081"],
        ["a=http://localhost", "a=http://elsewhere"],
        ["a=file:///tmp", "b=http://b"],
        ["a=http://u:p@a", "b=http://b"],
    ):
        with pytest.raises(ValueError):
            server.cameras(items)
    with pytest.raises(ValueError):
        server.checked_name("../record")


def test_native_sync_known_offset(tmp_path):
    source = make_video(tmp_path / "fixture")
    folder = tmp_path / "take"
    raw = folder / "raw_videos"
    raw.mkdir(parents=True)
    shutil.copyfile(source, raw / "test.id-cam1.idx-0.mkv")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "trim=start=0.4,setpts=PTS-STARTPTS",
            "-af",
            "atrim=start=0.4,asetpts=PTS-STARTPTS",
            "-c:v",
            "mjpeg",
            "-threads",
            "1",
            "-q:v",
            "3",
            "-c:a",
            "pcm_s16le",
            str(raw / "test.id-cam2.idx-0.mkv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "visual_mocap_server.py"),
            "--sync-job",
            str(folder),
        ],
        check=True,
        timeout=150,
    )
    result = json.loads((folder / "sync_result.json").read_text())
    lags = {v["video_name"]: v["lag_seconds"] for v in result["lags"]}
    assert sorted(lags.values()) == pytest.approx([0, 0.4], abs=0.07), lags
    assert len(result["videos_after"]) == 2 and result["synchronized_frame_count"] > 80
    assert len({v["frame_count"] for v in result["videos_after"]}) == 1
    import cv2

    frames = []
    for item in result["videos_after"]:
        capture = cv2.VideoCapture(item["filepath"])
        ok, frame = capture.read()
        capture.release()
        assert ok
        frames.append(frame)
    assert abs(frames[0].astype(float) - frames[1].astype(float)).mean() < 3
    assert server.synchronize(folder) == result


def test_native_sync_rejects_silence(tmp_path):
    source = make_video(tmp_path / "fixture", silent=True)
    raw = tmp_path / "take" / "raw_videos"
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        shutil.copyfile(source, raw / f"{name}.mkv")
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "visual_mocap_server.py"),
            "--sync-job",
            str(raw.parent),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0 and "silent" in result.stderr
    assert not (raw.parent / "sync_result.json").exists()


def test_two_nodes_record_manual_upload_and_calibration_replacement(tmp_path):
    source = make_video(tmp_path / "fixture")
    with (
        node_server(tmp_path, source, "cam1") as (one, base1),
        node_server(tmp_path, source, "cam2") as (two, base2),
    ):
        with host_server(tmp_path / "host", {"cam1": base1, "cam2": base2}) as (
            host,
            base,
        ):
            with request(base, "/") as response:
                assert "SkellySync" in response.read().decode()
            with request(
                base, "/api/start", dict(name="Доска 01", kind="calibration")
            ) as response:
                take = json.load(response)
            identifier = take["id"]
            wait_for(lambda: host.takes[identifier]["state"] == "recording")
            time.sleep(1)
            with request(base, "/api/stop", dict(id=identifier, upload=False)):
                pass
            wait_for(lambda: host.takes[identifier]["state"] in ("stopped", "failed"))
            assert host.takes[identifier]["state"] == "stopped", host.takes[identifier]
            host.worker.join(5)
            manifests = json.loads(
                (host.root / "takes" / identifier / "camera_manifests.json").read_text()
            )
            # Simulate an interrupted transfer. Real HTTP Range resumes this prefix.
            raw = host.root / "takes" / identifier / "raw_videos"
            raw.mkdir()
            full = one.root / identifier / "video.mkv"
            (raw / "Доска 01.id-cam1.idx-0.partial").write_bytes(
                full.read_bytes()[:1111]
            )
            with request(base, "/api/upload", dict(id=identifier)):
                pass
            wait_for(
                lambda: host.takes[identifier]["state"] in ("ready", "failed"),
                timeout=150,
            )
            assert host.takes[identifier]["state"] == "ready", host.takes[identifier]
            assert (
                server.digest(raw / "Доска 01.id-cam1.idx-0.mkv")
                == manifests["cam1"]["sha256"]
            )
            assert full.exists() and (two.root / identifier / "video.mkv").exists()
            assert host.calibration["id"] == identifier
            host.worker.join(5)
            with pytest.raises(HTTPError) as error:
                request(base, "/api/start", dict(name="Доска 02", kind="calibration"))
            assert error.value.code == 409
            with pytest.raises(HTTPError):
                request(
                    base,
                    "/api/start",
                    dict(
                        name="Доска 02",
                        kind="calibration",
                        replace_calibration=True,
                        expected_calibration_id="wrong",
                    ),
                )
            with request(
                base,
                "/api/start",
                dict(
                    name="Доска 02",
                    kind="calibration",
                    replace_calibration=True,
                    expected_calibration_id=identifier,
                ),
            ) as response:
                second = json.load(response)["id"]
            wait_for(lambda: host.takes[second]["state"] == "recording")
            assert (
                host.calibration["id"] == identifier
            )  # old set remains until new files synchronize
            with request(base, "/api/stop", dict(id=second)):
                pass
            wait_for(
                lambda: host.takes[second]["state"] in ("ready", "failed"), timeout=150
            )
            assert host.takes[second]["state"] == "ready", host.takes[second]
            wait_for(lambda: host.calibration["id"] == second)
            assert Path(host.takes[identifier]["synchronized_path"]).is_dir()


def test_start_failure_stops_other_camera(tmp_path):
    source = make_video(tmp_path / "fixture")
    with (
        node_server(tmp_path, source) as (node, base1),
        host_server(
            tmp_path / "host", {"cam1": base1, "cam2": "http://127.0.0.1:1"}
        ) as (host, base),
    ):
        with request(
            base, "/api/start", dict(name="failure", kind="record")
        ) as response:
            identifier = json.load(response)["id"]
        wait_for(lambda: host.takes[identifier]["state"] == "failed")
        wait_for(lambda: node.active is None)
        assert not host.calibration
        assert node.takes[identifier]["state"] == "ready"


def test_host_restart_marks_pending_take_interrupted(tmp_path):
    identifier = uuid.uuid4().hex
    folder = tmp_path / "takes" / identifier
    folder.mkdir(parents=True)
    server.save(
        folder / "take.json",
        dict(id=identifier, name="restart", state="uploading", created=time.time()),
    )
    with host_server(
        tmp_path, {"a": "http://127.0.0.1:1", "b": "http://127.0.0.1:2"}
    ) as (host, base):
        assert host.takes[identifier]["state"] == "interrupted"
        assert json.loads((folder / "take.json").read_text())["state"] == "interrupted"


def test_mkv_audio_tail_does_not_inflate_video_frame_count(tmp_path):
    source = make_video(tmp_path / "fixture")
    raw = tmp_path / "take" / "raw_videos"
    raw.mkdir(parents=True)
    shutil.copyfile(source, raw / "long.mkv")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "trim=end=3",
            "-c:v",
            "mjpeg",
            "-threads",
            "1",
            "-c:a",
            "copy",
            str(raw / "short_video_long_audio.mkv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "visual_mocap_server.py"),
            "--sync-job",
            str(raw.parent),
        ],
        check=True,
        timeout=90,
    )
    result = json.loads((raw.parent / "sync_result.json").read_text())
    assert result["synchronized_frame_count"] == 45
    assert {item["frame_count"] for item in result["videos_after"]} == {45}
