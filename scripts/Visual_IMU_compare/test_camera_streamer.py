"""Headless sender tests; physical-camera validation is performed separately."""
import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest

spec = importlib.util.spec_from_file_location("camera_streamer", Path(__file__).with_name("camera_streamer.py"))
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)

FORMATS = """
    [0]: 'YUYV' (YUYV 4:2:2)
        Size: Discrete 1280x720
            Interval: Discrete 0.100s (10.000 fps)
        Size: Discrete 640x480
            Interval: Discrete 0.033s (30.000 fps)
    [1]: 'MJPG' (Motion-JPEG, compressed)
        Size: Discrete 1280x720
            Interval: Discrete 0.033s (30.000 fps)
            Interval: Discrete 0.067s (15.000 fps)
    [2]: 'H264' (H.264)
        Size: Discrete 1280x720
            Interval: Discrete 0.033s (29.970 fps)
"""


def options(*extra):
    return s.arguments(["--server", "192.168.1.10", "--resolution", "1280x720", "--fps", "30", *extra])


def test_mode_selection_does_not_downgrade_fps():
    modes = s.parse_modes(FORMATS)
    assert s.choose_format(options(), modes) == "mjpeg"
    with pytest.raises(ValueError, match="does not advertise"):
        s.choose_format(options("--input-format", "yuyv422"), modes)
    with pytest.raises(ValueError, match="does not advertise"):
        s.choose_format(options("--fps", "60"), modes)


def test_fractional_fps_and_h264_passthrough():
    args = options("--fps", "30000/1001", "--encoder", "copy")
    assert s.choose_format(args, s.parse_modes(FORMATS)) == "h264"
    command = s.ffmpeg_command(args, "/usr/bin/ffmpeg", Path("/dev/video7"), "h264")
    assert command[command.index("-framerate")+1] == "30000/1001"
    assert command[command.index("-c:v")+1] == "copy"
    assert "-preset" not in command and "-pix_fmt" not in command


def test_real_capture_parameters_and_no_artificial_cfr():
    args = options("--name", "left", "--bitrate", "2M")
    command = s.ffmpeg_command(args, "/usr/bin/ffmpeg", Path("/dev/video1"), "mjpeg")
    assert command[command.index("-video_size")+1] == "1280x720"
    assert command[command.index("-framerate")+1] == "30"
    assert command[command.index("-input_format")+1] == "mjpeg"
    assert command[-1] == "rtsp://192.168.1.10:8554/left"
    assert "-r" not in command and "-vf" not in command
    assert command[command.index("-vsync")+1] == "0"
    assert "zerolatency" in command and "-an" in command


@pytest.mark.parametrize("extra", [
    ["--resolution", "1279x720"], ["--fps", "0"], ["--fps", "NaN"],
    ["--server", "rtsp://localhost"], ["--server", "host;touch /tmp/x"],
    ["--name", "../camera"], ["--port", "65536"], ["--duration", "nan"],
    ["--encoder", "copy", "--input-format", "mjpeg"],
])
def test_invalid_options_rejected(extra):
    with pytest.raises(SystemExit):
        options(*extra)


def test_required_parameters_and_relay_mode():
    with pytest.raises(SystemExit):
        s.arguments(["--server", "localhost"])
    assert s.arguments(["--relay"]).port == 8554
    assert s.rtsp_url("::1", 8554, "cam1") == "rtsp://[::1]:8554/cam1"
    text = s.relay_config("::1", 8554)
    assert '"[::1]:8554"' in text
    assert "overridePublisher: false" in text
    assert "hls: no" in text and "webrtc: no" in text


def test_supervisor_drains_both_pipes():
    code = "import sys\nfor i in range(1,5):\n print('warning '+'.'*4000,file=sys.stderr,flush=True)\n print('frame='+str(i)+'\\nfps=30\\nprogress=continue',flush=True)"
    result, count = s.stream_attempt([sys.executable, "-u", "-c", code], threading.Event(), time.monotonic()+5, 2)
    assert result == 0 and count == 4


def test_supervisor_restarts_when_frame_counter_stops():
    code = "import time\nwhile True:\n print('frame=1\\nprogress=continue',flush=True)\n time.sleep(.03)"
    started = time.monotonic()
    result, count = s.stream_attempt([sys.executable, "-u", "-c", code], threading.Event(), started+5, .25)
    assert result != 0 and count == 1
    assert time.monotonic()-started < 2


def test_wall_clock_duration_stops_child_without_frames():
    started = time.monotonic()
    result, count = s.stream_attempt([sys.executable, "-c", "import time; time.sleep(30)"],
                                    threading.Event(), started+.2, 5)
    assert count == 0 and result != 0
    assert time.monotonic()-started < 2
