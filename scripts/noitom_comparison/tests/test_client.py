"""Exercise the C boundary using pointer-writing SDK doubles, without a DLL."""
import ctypes as C
from dataclasses import replace
import json
import numpy as np
from scripts.noitom_comparison.config import ComparisonConfig
from scripts.noitom_comparison.noitom_client import (
    FakeNoitomClient, NoitomClient, NoitomReceiverThread, _Event,
)
from scripts.noitom_comparison.recording import ExperimentRecorder
from scripts.noitom_comparison.noitom_adapter import sdk_rotation_to_project


def sdk_double():
    client = NoitomClient(ComparisonConfig())
    source = FakeNoitomClient.frame(4, 1)
    handles = {name: index + 100 for index, name in enumerate(source.joints)}
    names = {value: name for name, value in handles.items()}
    state = {"index": 71, "source": source}

    def write(pointer, ctype, value):
        C.cast(pointer, C.POINTER(ctype))[0] = value

    def name(pointer, handle):
        write(pointer, C.c_char_p, names[handle].encode())
        return 0

    def tag(pointer, handle):
        write(pointer, C.c_int, state["source"].joints[names[handle]].tag)
        return 0

    def vector(attribute, pointers, handle):
        values = getattr(state["source"].joints[names[handle]], attribute)
        if attribute == "rotation":
            values = values[[1, 2, 3, 0]]  # SDK returns xyzw.
        for pointer, value in zip(pointers, values, strict=True):
            write(pointer, C.c_float, value)
        return 0

    def children(output, count, handle):
        children = [handles[key] for key, joint in state["source"].joints.items() if joint.parent == names[handle]]
        if output is not None:
            for index, value in enumerate(children):
                output[index] = value
        write(count, C.c_uint32, len(children))
        return 0

    def scalar(ctype, value):
        def call(pointer, handle):
            write(pointer, ctype, value() if callable(value) else value)
            return 0
        return call

    client._functions = {
        "avatar": {"index": scalar(C.c_uint32, 0), "root": scalar(C.c_uint64, handles["Hips"]),
                   "name": scalar(C.c_char_p, b"Actor"), "posture_index": scalar(C.c_uint32, lambda: state["index"])},
        "joint": {"name": name, "tag": tag, "children": children,
                  "rotation": lambda x, y, z, w, h: vector("rotation", (x, y, z, w), h),
                  "position": lambda x, y, z, h: vector("position", (x, y, z), h),
                  "default_position": lambda x, y, z, h: vector("default_position", (x, y, z), h)},
    }
    event = _Event()
    event.size, event.event_type, event.timestamp, event.data.avatar = C.sizeof(_Event), 256, 12.34, 9
    return client, state, event


def test_snapshots_all_joint_values_before_sdk_changes():
    client, state, event = sdk_double()
    client._queue_snapshot(event, 10001, None)
    state["index"] = 72
    state["source"] = FakeNoitomClient.frame(23, 8)
    client._queue_snapshot(event, 20002, None)
    first, second = client.poll_events()[0], client.poll_events()[0]
    assert (first.host_timestamp_ns, second.host_timestamp_ns) == (10001, 20002)
    assert (first.frame_index, second.frame_index) == (71, 72)
    assert first.sdk_timestamp == 12.34
    assert first.avatar_name == "Actor"
    assert first.joints["LeftHand"].parent == "LeftForeArm"
    np.testing.assert_allclose(first.joints["LeftArm"].rotation,
        sdk_rotation_to_project(FakeNoitomClient.frame(4, 0).joints["LeftArm"].rotation[[1, 2, 3, 0]]), atol=1e-7)
    np.testing.assert_allclose(first.joints["LeftArm"].sdk_rotation_xyzw,
        FakeNoitomClient.frame(4, 0).joints["LeftArm"].rotation[[1, 2, 3, 0]], atol=1e-7)
    assert not np.allclose(first.joints["LeftArm"].rotation, second.joints["LeftArm"].rotation)


def test_polled_event_receipt_precedes_joint_queries():
    client, state, event = sdk_double()
    order = []
    def poll(output, count, app):
        if output is None:
            order.append("count")
            C.cast(count, C.POINTER(C.c_uint32))[0] = 1
            return 0
        order.append("poll")
        output[0] = event
        C.cast(count, C.POINTER(C.c_uint32))[0] = 1
        return 0
    def stamp():
        order.append("stamp")
        return 3333, None
    original = client._functions["avatar"]["root"]
    def root(*args):
        order.append("snapshot")
        return original(*args)
    client._functions["application"] = {"poll": poll}
    client._functions["avatar"]["root"] = root
    client._opened, client.stamp = True, stamp
    frame = client.poll_events()[0]
    assert order == ["count", "poll", "stamp", "snapshot"]
    assert frame.host_timestamp_ns == 3333


def test_unsupported_default_offset_does_not_abort_live_bvh_frame():
    client, state, event = sdk_double()
    client._functions["joint"]["default_position"] = lambda *args: 6
    client._queue_snapshot(event, 123, None)
    frame = client.poll_events()[0]
    assert len(frame.joints) == len(state["source"].joints)
    assert all(joint.default_position is None for joint in frame.joints.values())
    assert frame.joints["LeftHand"].position is not None


def test_native_buffer_growth_requeries_count_and_keeps_frame():
    client, state, event = sdk_double()
    calls = []
    def poll(output, count, app):
        calls.append("count" if output is None else "fetch")
        if output is None:
            C.cast(count, C.POINTER(C.c_uint32))[0] = 1
            return 0
        if len(calls) == 2:
            return 2
        output[0] = event
        C.cast(count, C.POINTER(C.c_uint32))[0] = 1
        return 1  # Valid returned event with MoreEvent must not be dropped.
    client._functions["application"] = {"poll": poll}
    client._opened = True
    assert client.poll_events()[0].frame_index == 71
    assert calls == ["count", "fetch", "count", "fetch"]


def test_index_gap_on_inflight_frame_marks_original_session_incomplete(tmp_path):
    recorder = ExperimentRecorder()
    worker = NoitomReceiverThread(FakeNoitomClient(), recorder)
    session = recorder.start(tmp_path, {})
    for index in (100, 102):
        timestamp, claim = recorder.stamp_packet()
        frame = replace(FakeNoitomClient.frame(index, timestamp), capture_context=claim)
        if index == 102:
            recorder.stop()
        worker._consume(frame)
    assert session.done.wait(3)
    assert "noitom_frame_index_discontinuity" in session.metadata["incomplete_reasons"]
    saved = [json.loads(line) for line in (session.path / "noitom_frames.jsonl").read_text().splitlines()]
    assert [row["sdk_frame_index"] for row in saved] == [100, 102]
