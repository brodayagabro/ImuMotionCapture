"""Check boot-service generation without installing a service or opening a camera."""
import importlib.util
from pathlib import Path
import pwd
import shutil
import subprocess
import sys

import pytest


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "camera_streamer" not in sys.modules:
    load_module("camera_streamer")
if "skellycam_node" not in sys.modules:
    load_module("skellycam_node")
s = load_module("install_service")
ARGS = ["--server", "192.168.1.90", "--name", "cam2", "--device", "/dev/video987",
        "--resolution", "640x480", "--fps", "30000/1001", "--preview", "--preview-port", "8091"]
RECORDING_ARGS = ["--name", "cam2", "--device", "/dev/v4l/by-id/disconnected-camera-video-index0",
                  "--resolution", "1280x720", "--fps", "30", "--host", "0.0.0.0", "--port", "8082",
                  "--records", "runtime/takes %n $HOME", "--exposure-mode", "MANUAL", "--exposure", "-7"]
ACCOUNT = pwd.struct_passwd(("camera-user", "x", 1000, 1000, "", "/home/camera-user", "/bin/bash"))


@pytest.fixture
def camera_dir(tmp_path):
    directory = tmp_path / 'camera space %n ${HOME} "quote"'
    (directory / ".venv/bin").mkdir(parents=True)
    (directory / ".venv/bin/python").symlink_to(sys.executable)
    (directory / "camera_streamer.py").write_text("pass\n")
    (directory / ".venv-skellycam/bin").mkdir(parents=True)
    (directory / ".venv-skellycam/bin/python").symlink_to(sys.executable)
    (directory / "skellycam_node.py").write_text("pass\n")
    return directory


@pytest.mark.parametrize("workflow", ["live", "recording"])
def test_real_systemd_parser_and_venv_preserved(camera_dir, tmp_path, workflow):
    args = RECORDING_ARGS if workflow == "recording" else ARGS
    environment = ".venv-skellycam" if workflow == "recording" else ".venv"
    name, unit = s.render_unit(camera_dir, ACCOUNT, args, workflow=workflow)
    assert s.unit_quote(camera_dir / environment / "bin/python") in unit
    assert s.unit_quote(f"PATH={camera_dir}/{environment}/bin:/usr/local/bin:/usr/bin:/bin") in unit
    assert name == "visual-mocap-camera-cam2.service"
    if workflow == "recording":
        assert s.unit_quote(camera_dir / "skellycam_node.py") in unit
        assert '"--port" "8082"' in unit and '"--exposure" "-7"' in unit
        assert '"runtime/takes %%n $HOME"' in unit
        assert '"/dev/v4l/by-id/disconnected-camera-video-index0"' in unit
        assert "camera_streamer.py" not in unit and "--server" not in unit
        assert "TimeoutStopSec=60\n" in unit
        assert "KillMode=mixed\n" in unit  # main process finalizes files before stopping workers
    else:
        assert '"30000/1001"' in unit and '"--preview-port" "8091"' in unit
        assert "KillMode=control-group\n" in unit
    assert "ConditionPathExists=" not in unit  # late USB must retry, not skip forever
    assert "Restart=always\n" in unit
    path = s.write_unit(tmp_path, name, unit)
    if not shutil.which("systemd-analyze"):
        pytest.skip("systemd-analyze is unavailable on this test machine")
    result = subprocess.run(["systemd-analyze", "verify", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("extra", [["--list-devices"], ["--list-formats"], ["--duration", "10"]])
def test_one_shot_commands_are_rejected(camera_dir, extra):
    with pytest.raises(ValueError, match="boot service"):
        s.render_unit(camera_dir, ACCOUNT, [*ARGS, *extra])


@pytest.mark.parametrize("extra", [["--resolution", "641x480"], ["--name", "../bad"], ["--fps", "0"]])
def test_streamer_validation_applies(camera_dir, extra):
    with pytest.raises(SystemExit):
        s.render_unit(camera_dir, ACCOUNT, [*ARGS, *extra])


def test_missing_environment_and_unit_injection_rejected(camera_dir):
    with pytest.raises(ValueError, match="control characters"):
        s.render_unit(camera_dir, ACCOUNT, [*ARGS, "--device", "/dev/video0\nUser=0"])
    (camera_dir / ".venv/bin/python").unlink()
    with pytest.raises(ValueError, match="setup.sh"):
        s.render_unit(camera_dir, ACCOUNT, ARGS)


def test_generate_only_and_repeat_replace_settings(camera_dir, monkeypatch):
    monkeypatch.setattr(s, "CAMERA_DIR", camera_dir)
    monkeypatch.setattr(s, "service_account", lambda username: ACCOUNT)
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **kw: pytest.fail("Generation must not start subprocesses"))
    assert s.main(ARGS) == 0
    path = camera_dir / "runtime/systemd/visual-mocap-camera-cam2.service"
    assert '"640x480"' in path.read_text()
    assert s.main([*ARGS, "--resolution", "1280x720"]) == 0
    assert '"1280x720"' in path.read_text()
    assert list(path.parent.iterdir()) == [path]


def test_switch_workflow_replaces_and_installs_the_same_service(camera_dir, monkeypatch):
    monkeypatch.setattr(s, "CAMERA_DIR", camera_dir)
    monkeypatch.setattr(s, "service_account", lambda username: ACCOUNT)
    installed = []
    monkeypatch.setattr(s, "install_unit", lambda path: installed.append((path, path.read_text())))
    assert s.main(["--install", *ARGS]) == 0
    path = camera_dir / "runtime/systemd/visual-mocap-camera-cam2.service"
    assert "camera_streamer.py" in path.read_text()
    assert s.main(["--workflow", "recording", "--install", *RECORDING_ARGS]) == 0
    assert installed[-1][0] == installed[0][0] == path
    assert "skellycam_node.py" in installed[-1][1] and "camera_streamer.py" not in installed[-1][1]
    assert list(path.parent.iterdir()) == [path]
    # Explicit live mode restores RTSP using the same unit, preserving old commands.
    assert s.main(["--workflow", "live", "--install", *ARGS]) == 0
    assert installed[-1][0] == path and "camera_streamer.py" in installed[-1][1]


@pytest.mark.parametrize("extra", [
    ["--resolution", "641x480"], ["--resolution", "640*480"], ["--name", "../bad"],
    ["--fps", "0"], ["--fps", "nan"], ["--fps", "inf"], ["--port", "0"], ["--port", "65536"],
    ["--server", "192.168.1.90"], ["--preview"],
])
def test_recording_arguments_fail_before_replacing_service(camera_dir, monkeypatch, extra):
    monkeypatch.setattr(s, "CAMERA_DIR", camera_dir)
    monkeypatch.setattr(s, "service_account", lambda username: ACCOUNT)
    monkeypatch.setattr(s, "install_unit", lambda path: pytest.fail("Invalid settings must not install a service"))
    assert s.main(ARGS) == 0
    path = camera_dir / "runtime/systemd/visual-mocap-camera-cam2.service"
    original = path.read_text()
    with pytest.raises(SystemExit):
        s.main(["--workflow", "recording", "--install", *RECORDING_ARGS, *extra])
    assert path.read_text() == original


def test_recording_missing_venv_does_not_fall_back_or_replace_service(camera_dir, monkeypatch):
    monkeypatch.setattr(s, "CAMERA_DIR", camera_dir)
    monkeypatch.setattr(s, "service_account", lambda username: ACCOUNT)
    monkeypatch.setattr(s, "install_unit", lambda path: pytest.fail("Missing venv must not install a service"))
    assert s.main(ARGS) == 0
    path = camera_dir / "runtime/systemd/visual-mocap-camera-cam2.service"
    original = path.read_text()
    (camera_dir / ".venv-skellycam/bin/python").unlink()
    assert s.main(["--workflow", "recording", "--install", *RECORDING_ARGS]) == 1
    assert path.read_text() == original


def test_recording_arguments_reject_unit_injection(camera_dir):
    with pytest.raises(ValueError, match="control characters"):
        s.render_unit(camera_dir, ACCOUNT, [*RECORDING_ARGS, "--device", "/dev/video0\nUser=0"], workflow="recording")


def test_root_is_not_selected_for_capture(monkeypatch):
    monkeypatch.setattr(s.os, "getuid", lambda: 0)
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setattr(s.pwd, "getpwuid", lambda uid: pwd.struct_passwd(("root", "x", 0, 0, "", "/root", "/bin/bash")))
    with pytest.raises(ValueError, match="non-root"):
        s.service_account()
    monkeypatch.setenv("SUDO_USER", "camera-user")
    monkeypatch.setattr(s.pwd, "getpwnam", lambda username: ACCOUNT)
    monkeypatch.setattr(s.grp, "getgrnam", lambda name: object())
    assert s.service_account() == ACCOUNT


def test_install_requests_restart_for_existing_service_and_stops_on_error(tmp_path, monkeypatch):
    unit_file = tmp_path / "visual-mocap-camera-cam2.service"
    monkeypatch.setattr(s.Path, "is_dir", lambda path: True)
    monkeypatch.setattr(s.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(s.os, "geteuid", lambda: 1000)
    calls = []

    def run(command, check):
        assert check
        calls.append(command)

    monkeypatch.setattr(s.subprocess, "run", run)
    s.install_unit(unit_file)
    assert all(command[0] == "sudo" for command in calls)
    assert calls[-2] == ["sudo", "systemctl", "enable", unit_file.name]
    assert calls[-1] == ["sudo", "systemctl", "restart", unit_file.name]
    calls.clear()

    def fail(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(s.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        s.install_unit(unit_file)
    assert len(calls) == 1  # no reload/enable/restart if copying failed
