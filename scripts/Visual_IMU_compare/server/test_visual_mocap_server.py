"""Geometry, BVH and network tests; no physical cameras or GPU required."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location("visual_mocap_server", Path(__file__).with_name("visual_mocap_server.py"))
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


@pytest.fixture
def args():
    return s.arguments(["--demo", "--device", "cpu", "--bvh-warmup", "1"])


@pytest.fixture
def geometry(args):
    ids = ["cam1", "cam2", "cam3"]
    cameras = s.checked_calibration(s.demo_calibration(ids), ids)
    return cameras, s.load_triangulator(args.repos)


def test_triangulation_with_distortion_outlier_and_missing_joint(geometry):
    cameras, triangulate = geometry
    for c in cameras.values():
        c["D"] = np.array([.07, -.03, .002, -.001, 0])
    expected = s.demo_points(.8)
    observations = {name: (s.project(expected, c), np.ones(17)) for name, c in cameras.items()}
    observations["cam3"][0][7] += [120, -95]
    for points, scores in observations.values():
        scores[9] = 0
    result = s.reconstruct(observations, cameras, triangulate)
    assert result["points"][9] is None
    valid = [i for i in range(17) if i != 9]
    np.testing.assert_allclose(np.array([result["points"][i] for i in valid]), expected[valid], atol=1e-5)


def test_single_view_and_nearly_parallel_rays_are_not_3d(geometry):
    cameras, triangulate = geometry
    expected = s.demo_points(0)
    observations = {"cam1": (s.project(expected, cameras["cam1"]), np.ones(17))}
    assert all(p is None for p in s.reconstruct(observations, cameras, triangulate)["points"])
    cameras["cam2"] = cameras["cam1"]
    observations["cam2"] = observations["cam1"]
    assert all(p is None for p in s.reconstruct(observations, cameras, triangulate)["points"])


def test_invalid_calibration_is_rejected():
    data = s.demo_calibration(["a", "b"])
    data["cameras"]["a"]["R"][0][0] = 8
    with pytest.raises(ValueError, match="rotation"):
        s.checked_calibration(data, ["a", "b"])


def test_buffer_selection_and_stalled_camera():
    def f(seq, t):
        return s.Frame(seq, t, t, np.empty((1, 1, 3)))
    histories = {"a": [f(1, 9.93), f(2, 9.97)], "b": [f(1, 9.94)], "stale": [f(1, 8)]}
    group = s.select_frames(histories, 10, .5, .05)
    assert set(group) == {"a", "b"}
    assert group["a"].sequence == 2
    assert s.select_frames(histories, 12, .5, .05) == {}
    histories = {"a": [f(1, 9.7), f(2, 9.99)], "b": [f(1, 9.7), f(2, 9.98)], "slow": [f(1, 9.7)]}
    group = s.select_frames(histories, 10, .5, .05)
    assert set(group) == {"a", "b"} and group["a"].sequence == 2


def test_model_does_not_invent_people_on_empty_detection():
    from types import SimpleNamespace
    model = object.__new__(s.PoseModel)
    model.body = SimpleNamespace(det_model=lambda image: [], pose_model=lambda *a, **kw: pytest.fail("pose must not run"))
    _, scores, people = model.infer(s.Frame(1, 1, 1, np.zeros((50, 50, 3))))
    assert people == 0 and not scores.any()
    model.body.det_model = lambda image: [[0, 0, 10, 10], [20, 20, 30, 30]]
    _, scores, people = model.infer(s.Frame(2, 2, 2, np.zeros((50, 50, 3))))
    assert people == 2 and not scores.any()


def rotation_from_channels(values):
    z, x, y = np.radians(values)
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry


def test_bvh_root_motion_forward_kinematics_missing_limb_and_bounded_memory(args):
    adapter = s.BvhAdapter(args)
    points = s.demo_points(.4)
    frame = adapter.update({"points": points.tolist()}, 1, 1)
    assert len(frame["channels"]) == len(adapter.metadata["channel_names"]) == 51
    expected_root = adapter.axis @ ((points[11]+points[12])/2) * 100
    np.testing.assert_allclose(frame["channels"][:3], expected_root, atol=1e-5)
    rotations, positions = [], []
    for i, joint in enumerate(adapter.recording.joints):
        local = rotation_from_channels(frame["channels"][3+i*3:6+i*3])
        if joint.parent is None:
            rotations.append(local)
            positions.append(np.array(frame["channels"][:3])/100)
        else:
            parent = joint.parent
            rotations.append(rotations[parent] @ local)
            offset = adapter.module.TO_BVH @ np.array(joint.offset)
            positions.append(positions[parent] + rotations[parent] @ offset)
    # Endpoints reconstruct measured directions in the fixed bone hierarchy.
    for a, b, start, end in [(4,5,5,7),(5,6,7,9),(10,11,11,13),(11,12,13,15)]:
        actual = positions[b]-positions[a]
        expected = adapter.axis @ (points[end]-points[start])
        np.testing.assert_allclose(actual/np.linalg.norm(actual), expected/np.linalg.norm(expected), atol=1e-6)
    translated = points + [1, 0, .3]
    second = adapter.update({"points": translated.tolist()}, 1.1, 2)
    np.testing.assert_allclose(np.array(second["channels"][:3])-frame["channels"][:3], [100, 0, -30], atol=1e-5)
    missing = translated.tolist()
    missing[9] = None
    third = adapter.update({"points": missing}, 1.2, 3)
    assert third["tracked"][5] is False
    assert np.isfinite(third["channels"]).all()
    assert len(adapter.recording.frames) == 1
    missing[11] = None
    assert adapter.update({"points": missing}, 1.3, 4) is None


def test_bvh_file_matches_channel_count_and_uniform_grid(args, tmp_path):
    args.bvh_file = tmp_path / "motion.bvh"
    adapter = s.BvhAdapter(args)
    for i, timestamp in enumerate([10., 10.12, 10.25]):
        adapter.update({"points": s.demo_points(i*.1).tolist()}, timestamp, i)
    adapter.close()
    text = args.bvh_file.read_text()
    header, motion = text.split("MOTION\n")
    assert header.startswith("HIERARCHY\nROOT Hips")
    lines = motion.splitlines()
    count = int(lines[0].split(":")[1])
    assert count == 6 == len(lines[2:])
    assert all(len(line.split()) == 51 for line in lines[2:])
    assert not args.bvh_file.with_name("motion.bvh.part").exists()


@pytest.mark.parametrize("transport", ["udp", "tcp"])
def test_bvh_transport_roundtrip(args, transport):
    kind = socket.SOCK_DGRAM if transport == "udp" else socket.SOCK_STREAM
    receiver = socket.socket(socket.AF_INET, kind)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(3)
    if transport == "tcp":
        receiver.listen(1)
    stop = threading.Event()
    sender = s.BvhSender(f"127.0.0.1:{receiver.getsockname()[1]}", transport, stop)
    adapter = s.BvhAdapter(args)
    frame = adapter.update({"points": s.demo_points(0).tolist()}, 1, 1)
    sender.thread.start()
    connection = None
    try:
        sender.submit(adapter.metadata, frame)
        if transport == "udp":
            messages = [json.loads(receiver.recvfrom(65535)[0]) for _ in range(2)]
        else:
            connection, _ = receiver.accept()
            connection.settimeout(3)
            with connection.makefile("rb") as stream:
                messages = [json.loads(stream.readline()) for _ in range(2)]
        assert [m["type"] for m in messages] == ["skeleton", "frame"]
        assert messages[0]["skeleton_id"] == messages[1]["skeleton_id"]
        assert messages[1]["channels"] == frame["channels"]
    finally:
        stop.set()
        sender.wake.set()
        sender.thread.join(2)
        if connection:
            connection.close()
        receiver.close()
    assert not sender.thread.is_alive()


def test_calibration_solver_recovers_known_stereo_geometry(args):
    session = s.CalibrationSession(args)
    objects = session.board.getChessboardCorners().astype(float)
    k = np.array([[900.,0,640],[0,900,360],[0,0,1]])
    r, _ = s.cv2.Rodrigues(np.array([.02,.17,-.01]))
    cameras = {"a":dict(K=k,D=np.zeros(5),R=np.eye(3),t=np.zeros(3)),
               "b":dict(K=k,D=np.zeros(5),R=r,t=np.array([-.6,.01,.02]))}
    for i in range(16):
        rotation, _ = s.cv2.Rodrigues(np.array([.3*np.sin(i),.35*np.cos(i*.8),.1*np.sin(i*.5)]))
        positions = (rotation @ objects.T).T + [-.2+.1*np.sin(i*.6),-.1+.08*np.cos(i),1.5+.2*np.sin(i*.3)]
        session.samples.append({name:dict(ids=np.arange(len(objects)), corners=s.project(positions,c).astype(np.float32),size=(1280,720)) for name,c in cameras.items()})
    result = session.solve(["a","b"])
    np.testing.assert_allclose(result["cameras"]["b"]["t"], cameras["b"]["t"], atol=.002)
    np.testing.assert_allclose(result["cameras"]["b"]["R"], r, atol=.002)
    assert max(result["rms_px"].values()) < .01


def test_websocket_demo_and_api(args):
    engine = s.Engine(args)
    with TestClient(s.create_app(engine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/calibration/board.png").headers["content-type"] == "image/png"
        assert client.get("/video/missing.mjpg").status_code == 404
        assert client.post("/api/calibration/capture").status_code == 409
        with client.websocket_connect("/ws") as ws:
            for _ in range(80):
                state = ws.receive_json()
                if state["pose"] and state["pose"]["bvh"]:
                    break
            assert state["demo"]
            assert sum(p is not None for p in state["pose"]["skeleton"]["points"]) == 17
            assert len(state["pose"]["bvh"]["channels"]) == 51
            assert client.get("/api/bvh/skeleton").json()["type"] == "skeleton"
            engine.stop.set()
            engine.thread.join(2)
            with engine.lock:
                engine.result = {"completed_monotonic_s":time.monotonic()-2}
                assert engine.result
            assert engine.snapshot()["pose"] is None
    assert not engine.thread.is_alive()
    assert all(not camera.thread.is_alive() for camera in engine.cameras.values())


def test_cuda_must_not_silently_fall_back_to_cpu(args, monkeypatch):
    import onnxruntime
    args.device = "cuda"
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider unavailable"):
        s.PoseModel(args)


def test_rejects_nonmonotonic_bvh_timestamps(args):
    adapter = s.BvhAdapter(args)
    pose = {"points":s.demo_points(0).tolist()}
    assert adapter.update(pose, 2, 1)
    assert adapter.update(pose, 1, 2) is None
    assert adapter.update(pose, 2, 3) is None


def test_charuco_board_is_detectable(args):
    session = s.CalibrationSession(args)
    cols, rows = session.board.getChessboardSize()
    image = session.board.generateImage((cols*160+80, rows*160+80), marginSize=40)
    _, ids, _, _ = session.detector.detectBoard(image)
    assert len(ids) == (cols-1)*(rows-1)


def test_idle_armature_exists_without_cameras_or_bvh(args):
    view = s.ArmatureView(.5)
    state = view.snapshot(100)
    assert state["status"] == "rest" and state["frame_id"] is None
    assert len(state["points"]) == 16 and len(state["bones"]) == 19
    assert min(p[1] for p in state["points"]) == pytest.approx(0)
    assert all(np.linalg.norm(np.subtract(b["end"], b["start"])) > 0 for b in state["bones"])
    engine = s.Engine(args)
    assert engine.snapshot()["armature"] == state
    # Idle geometry is embedded, so the browser can draw before its first WS message.
    assert '__REST_ARMATURE__' not in s.HTML


def test_armature_forward_kinematics_uses_parent_offsets_and_zxy_order():
    rig = [dict(name="root", parent=None, offset=(0,0,0), end=None),
           dict(name="child", parent=0, offset=(1,0,0), end=(0,1,0))]
    channels = [2,3,4, 30,40,50, -20,10,60]
    geometry = s.armature_geometry(rig, channels, [True, False])
    root = np.array(channels[:3])
    r0 = rotation_from_channels(channels[3:6])
    r1 = rotation_from_channels(channels[6:9])
    child = root+r0 @ [1,0,0]
    np.testing.assert_allclose(geometry["points"][1], child)
    np.testing.assert_allclose(geometry["bones"][1]["end"], child+r0 @ r1 @ [0,1,0])
    np.testing.assert_allclose(geometry["bones"][1]["axes"], (r0 @ r1).T)
    assert geometry["bones"][0]["tracked"] and not geometry["bones"][1]["tracked"]


@pytest.mark.parametrize("units", ["m", "cm"])
def test_armature_follows_exported_bvh_with_fixed_lengths(args, units):
    args.bvh_units = units
    adapter, view = s.BvhAdapter(args), s.ArmatureView(.5)
    for i in range(3):
        frame = adapter.update({"points":s.demo_points(i*.1).tolist()}, 1+i*.05, i)
        original_channels = frame["channels"][:]
        view.update(adapter, frame, 1.01+i*.05)
        assert frame["channels"] == original_channels
        assert view.snapshot(1.01+i*.05)["status"] == ("live" if i == 2 else "waiting")
    before = view.snapshot(1.11)
    scale = 100 if units == "cm" else 1
    np.testing.assert_allclose(before["points"][0], np.array(frame["channels"][:3])/scale)
    # Different pose / root translation / noisy estimated proportions must not stretch bones.
    points = s.demo_points(1.4)+[1,0,.3]
    points[9] += [.05,0,0]
    frame = adapter.update({"points":points.tolist()}, 1.2, 3)
    view.update(adapter, frame, 1.21)
    after = view.snapshot(1.21)
    assert not np.allclose(before["points"], after["points"])
    for a, b in zip(before["bones"], after["bones"], strict=True):
        assert a["name"] == b["name"]
        assert np.linalg.norm(np.subtract(a["end"],a["start"])) == pytest.approx(
            np.linalg.norm(np.subtract(b["end"],b["start"])))
    # Displayed forearm direction agrees with the measured, exported swing.
    actual = np.subtract(after["points"][6], after["points"][5])
    expected = adapter.axis @ (points[9]-points[7])
    np.testing.assert_allclose(actual/np.linalg.norm(actual), expected/np.linalg.norm(expected), atol=1e-6)


def test_armature_holds_loss_and_requires_fresh_sequence_to_resume(args):
    adapter, view = s.BvhAdapter(args), s.ArmatureView(.5)
    for i in range(3):
        frame = adapter.update({"points":s.demo_points(i).tolist()}, 1+i*.05, i)
        view.update(adapter, frame, 1.01+i*.05)
    good = view.snapshot(1.11)
    assert good["status"] == "live"
    # Expiration works even if inference stops producing updates completely.
    assert view.snapshot(2)["status"] == "held"
    view.update(adapter, None, 2)
    assert view.snapshot(2)["points"] == good["points"]
    for i in range(3):
        frame = adapter.update({"points":s.demo_points(i+.5).tolist()}, 2.1+i*.05, i+3)
        view.update(adapter, frame, 2.11+i*.05)
        state = view.snapshot(2.11+i*.05)
        assert state["status"] == ("live" if i == 2 else "held")
        if i < 2:
            assert state["points"] == good["points"]
    view.update(adapter, frame, 2.22)  # duplicate is not a new stable frame
    assert view.snapshot(2.22)["status"] == "held"
    assert view.consecutive == 0


def test_armature_rejects_stale_nonfinite_and_resets_for_new_bvh(args):
    adapter, view = s.BvhAdapter(args), s.ArmatureView(.5)
    frame = adapter.update({"points":s.demo_points(0).tolist()}, 1, 1)
    view.update(adapter, frame, 2)
    assert view.consecutive == 0
    broken = dict(frame, channels=[float("nan")]+frame["channels"][1:])
    view.update(adapter, broken, 1.1)
    assert view.consecutive == 0
    view.update(adapter, frame, 1.1)
    identity = view.identity
    replacement = s.BvhAdapter(args)
    new_frame = replacement.update({"points":s.demo_points(0).tolist()}, 2, 1)
    view.update(replacement, new_frame, 2.01)
    assert view.identity != identity
    assert view.consecutive == 1 and view.snapshot(2.01)["status"] == "waiting"
