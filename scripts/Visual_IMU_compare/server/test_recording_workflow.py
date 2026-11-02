"""Transport/state tests plus real SkellySync alignment (no sync mocks)."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import freemocap_host as host_module
from freemocap_recording_worker import synchronize, processing_timeline

@pytest.fixture(scope="module", autouse=True)
def release_unused_native_websocket_logs():
    yield
    # Native imports create a UI log queue. These headless tests have no UI
    # consumer; waiting for its feeder to flush a full pipe hangs interpreter
    # shutdown. Only discard this test process's unused websocket messages.
    from skellylogs.handlers.websocket_log_queue_handler import get_websocket_log_queue
    try:
        get_websocket_log_queue().cancel_join_thread()
    except ValueError:
        pass  # No native logging queue was created by the selected tests.


def make_video(path, flash=30, count=105, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((120, 160, 3), 230 if flash <= index < flash+6 else 30, np.uint8))
    writer.release()


def test_native_flash_alignment_and_report(tmp_path):
    for name, flash in [("cam1", 30), ("cam2", 45), ("cam3", 21)]:
        make_video(tmp_path / "raw_videos" / f"take.id-{name}.idx-0.mp4", flash)
    result = synchronize(tmp_path, 1000, lambda *args, **kwargs: None)
    data = json.loads((tmp_path / "sync_result.json").read_text())
    assert data["aligned_flash_frames"] == [21, 21, 21]
    assert result.synchronized_frame_count == 81
    assert sorted(lag.lag_seconds for lag in result.lags) == pytest.approx([0, .3, .8])
    assert len(list((tmp_path / "raw_videos").glob("*.mp4"))) == 3


def test_no_flash_is_rejected_instead_of_native_noise_fallback(tmp_path):
    for name in ("cam1", "cam2"):
        make_video(tmp_path / "raw_videos" / f"{name}.mp4", flash=1000)
    with pytest.raises(ValueError, match="no flash"):
        synchronize(tmp_path, 1000, lambda *args, **kwargs: None)
    assert not (tmp_path / "sync_result.json").exists()


def test_processing_timeline_preserves_video_fps_for_native_readers(tmp_path):
    make_video(tmp_path / "synchronized_videos/cam1.mp4", fps=25)
    make_video(tmp_path / "synchronized_videos/cam2.mp4", fps=25)
    processing_timeline(tmp_path)
    # Both native consumers must see 25 FPS, not filter default 30 / Blender default 24.
    import multiprocessing.queues
    from freemocap.core.tasks.mocap.mocap_helpers.recording_framerate import get_recording_framerate
    from freemocap_blender_addon.utilities.recording_framerate import get_recording_framerate as blender_fps
    assert get_recording_framerate(tmp_path) == pytest.approx(25)
    assert blender_fps(tmp_path) == pytest.approx(25)
    provenance = json.loads((tmp_path / "synchronized_videos/timestamps/provenance.json").read_text())
    assert provenance["hardware_capture_timestamps"] is False


def test_bvh_publication_is_atomic_and_never_overwrites(tmp_path):
    source = tmp_path / "source.bvh"
    source.write_text("native BVH")
    directory = tmp_path / "output"
    target = host_module.publish_bvh(source, str(directory), "motion.bvh")
    assert target.read_text() == "native BVH"
    source.write_text("changed")
    with pytest.raises(FileExistsError):
        host_module.publish_bvh(source, str(directory), "motion.bvh")
    assert target.read_text() == "native BVH"
    assert not list(directory.glob(".bvh-*"))
    with pytest.raises(ValueError):
        host_module.publish_bvh(source, str(directory), "../outside.bvh")


def test_native_normalizes_measured_camera_rates(tmp_path):
    make_video(tmp_path / "raw_videos/cam1.mp4", fps=30)
    make_video(tmp_path / "raw_videos/cam2.mp4", fps=25)
    result = synchronize(tmp_path, 1000, lambda *args, **kwargs: None)
    assert all(video.fps == pytest.approx(25) for video in result.videos_after)
    data = json.loads((tmp_path / "sync_result.json").read_text())
    # Native FFmpeg resampling may introduce a common leading-frame delay.
    # The required invariant is coincident events on the resulting timeline.
    assert len(set(data["aligned_flash_frames"])) == 1
    assert data["aligned_flash_frames"][0] > 0


def host_args(tmp_path):
    return SimpleNamespace(records=tmp_path / "host", output=tmp_path / "output", blender="blender",
                           camera=["cam1=rtsp://127.0.0.1:8554/cam1", "cam2=rtsp://127.0.0.1:8554/cam2"])


def test_web_routes_output_picker_and_restart_recovery(tmp_path):
    args = host_args(tmp_path)
    args.output.mkdir()
    (args.output / "experiment").mkdir()
    folder = args.records / "test"
    folder.mkdir(parents=True)
    host_module.save(folder / "take.json", dict(id="test", state="processing", kind="sync", created=1,
                                               cameras=host_module.camera_endpoints(args.camera)))
    app = host_module.create_app(args)
    assert app.state.host.takes["test"]["state"] == "interrupted"
    with TestClient(app) as client:
        assert "BVH" in client.get("/").text
        assert client.get("/assets/bvh_player.js").status_code == 200
        assert client.get("/assets/freemocap_host.py").status_code == 404
        directories = client.get("/api/directories", params={"path": str(args.output)}).json()
        assert str(args.output / "experiment") in directories["directories"]
        assert client.get("/api/recordings/test/bvh").status_code == 404


def test_preview_endpoint_handles_warmup_unknown_camera_and_jpeg(tmp_path, monkeypatch):
    app = host_module.create_app(host_args(tmp_path))
    preview = app.state.host.previews["cam1"]
    monkeypatch.setattr(preview, "snapshot", lambda: None)
    with TestClient(app) as client:
        assert client.get("/api/cameras/unknown/preview.jpg").status_code == 404
        response = client.get("/api/cameras/cam1/preview.jpg")
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        monkeypatch.setattr(preview, "snapshot", lambda: b"jpeg")
        response = client.get("/api/cameras/cam1/preview.jpg")
        assert response.status_code == 200 and response.content == b"jpeg"
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("settings", [
    {"camera_offsets_ms": {"unknown": 1}},
    {"camera_offsets_ms": {"cam1": float("nan")}},
    {"max_skew_ms": 100, "max_age_ms": 50},
])
def test_invalid_reception_settings_fail_before_starting_capture(tmp_path, settings):
    import asyncio
    host = host_module.Host(host_args(tmp_path))
    with pytest.raises(host_module.HTTPException, match="Invalid camera timing"):
        asyncio.run(host.start(host_module.StartRequest(kind="sync", **settings)))
    assert not host.takes and host.capture is None
