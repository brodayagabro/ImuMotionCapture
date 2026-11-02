"""Transport/state tests plus real SkellySync alignment (no sync mocks)."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import freemocap_host as host_module
from freemocap_recording_worker import synchronize, processing_timeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skellycam_node", ROOT / "camera/skellycam_node.py")
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


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


class FakeRecorder:
    def __init__(self, args):
        self.args = args
        self.ready = False
        self.active = None

    async def prepare(self):
        self.ready = True

    async def start(self, take_id):
        self.active = take_id

    async def stop(self):
        path = self.args.records / self.active / "synchronized_videos" / f"test.id-{self.args.name}.idx-0.mp4"
        make_video(path, 30 if self.args.name == "cam1" else 45)
        self.active = None

    async def close(self):
        pass


def node_args(tmp_path, name="cam1"):
    return SimpleNamespace(name=name, device="/dev/video0", records=tmp_path / name,
                           resolution="160x120", fps=30)


def test_camera_node_finalization_checksum_idempotence_and_safe_paths(tmp_path):
    args = node_args(tmp_path)
    with TestClient(node.create_app(args, FakeRecorder)) as client:
        assert client.post("/prepare").json()["ready"]
        assert client.post("/recordings/test/start").status_code == 200
        assert client.post("/recordings/test/start").status_code == 200
        assert client.get("/recordings/test/manifest").status_code == 409
        assert client.post("/recordings/other/start").status_code == 409
        assert client.post("/recordings/test/stop").status_code == 200
        assert client.post("/recordings/test/stop").status_code == 200
        item = client.get("/recordings/test/manifest").json()["files"][0]
        response = client.get("/recordings/test/files/" + item["path"])
        import hashlib
        assert hashlib.sha256(response.content).hexdigest() == item["sha256"]
        assert len(response.content) == item["size"]
        assert client.get("/recordings/test/files/.node-state.json").status_code == 404
        assert client.get("/recordings/test/files/%2E%2E/%2E%2E/secret").status_code == 404
        assert client.post("/recordings/test/start").status_code == 409


def host_args(tmp_path):
    return SimpleNamespace(records=tmp_path / "host", output=tmp_path / "output", blender="blender",
                           camera=["cam1=http://cam1:8082", "cam2=http://cam2:8082"])


def test_camera_endpoint_identity_validation():
    for endpoints in [["cam1=rtsp://foo", "cam2=http://bar"], ["cam1=http://foo", "cam1=http://bar"],
                      ["cam1=http://foo", "cam2=http://foo"]]:
        with pytest.raises(ValueError):
            host_module.camera_endpoints(endpoints)


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


def test_two_nodes_transfer_to_real_sync_worker(tmp_path):
    async def exercise():
        from contextlib import AsyncExitStack
        args = host_args(tmp_path)
        host = host_module.Host(args)
        async with AsyncExitStack() as stack:
            clients = {}
            for name in ("cam1", "cam2"):
                app = node.create_app(node_args(tmp_path, name), FakeRecorder)
                await stack.enter_async_context(app.router.lifespan_context(app))
                clients[name] = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url=f"http://{name}:8082"))

            async def request(req):
                response = await clients[req.url.host].request(req.method, str(req.url))
                return httpx.Response(response.status_code, content=response.content, headers=response.headers)

            host.client = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(request)))
            take = await host.start(host_module.StartRequest(kind="sync"))
            assert take["state"] == "recording"
            await host.stop(take["id"])
            await asyncio.wait_for(host.task, 45)
            assert take["state"] == "complete", take
            folder = host.root / take["id"]
            data = json.loads((folder / "sync_result.json").read_text())
            assert data["aligned_flash_frames"] == [30, 30]
            assert len(list((folder / "raw_videos").glob("*.mp4"))) == 2
            for name in clients:
                assert (tmp_path / name / take["id"]).is_dir()
            await host.close()
    asyncio.run(exercise())


def test_failed_start_rolls_back_every_node(tmp_path):
    async def exercise():
        host = host_module.Host(host_args(tmp_path))
        stopped = []
        async def request(req):
            if req.url.path == "/prepare":
                return httpx.Response(200, json=dict(camera=req.url.host, ready=True, active=None, fps=30))
            if req.url.path.endswith("/start") and req.url.host == "cam2":
                return httpx.Response(503, text="device busy")
            if req.url.path.endswith("/stop"):
                stopped.append(req.url.host)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            host.client = client
            with pytest.raises(host_module.HTTPException, match="device busy"):
                await host.start(host_module.StartRequest(kind="sync"))
        assert sorted(stopped) == ["cam1", "cam2"]
        assert next(iter(host.takes.values()))["state"] == "failed"
    asyncio.run(exercise())
