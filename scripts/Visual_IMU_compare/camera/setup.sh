#!/usr/bin/env bash
# Local, repeatable setup for the camera computer. Never activate another venv.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN=/usr/bin/python3
EXPLICIT_PYTHON=0
INSTALL_SYSTEM=0
WITH_TESTS=0

usage() {
    cat <<'EOF'
Usage: ./setup.sh [--python /path/to/python3] [--system-deps] [--with-tests]

Creates camera/.venv (Python >= 3.10), installs pinned dependencies and checks
FFmpeg + V4L2. Safe to repeat. No camera is opened and no service is started.
If system Python is too old, a private Python 3.12 is downloaded automatically.
--system-deps  Install python3-venv, ffmpeg, v4l-utils through apt/sudo.
--with-tests   Also install pytest.
EOF
}
fail() { echo "ERROR: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --python) (($# >= 2)) || fail "--python requires a value"; PYTHON_BIN=$2; EXPLICIT_PYTHON=1; shift 2 ;;
        --system-deps) INSTALL_SYSTEM=1; shift ;;
        --with-tests) WITH_TESTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ $(uname -s) == Linux ]] || fail "This profile is for Linux V4L2"
command -v flock >/dev/null || fail "flock (util-linux) is required"
mkdir -p -- "$SCRIPT_DIR/.tools"
exec 9>"$SCRIPT_DIR/.tools/setup.lock"
flock -n 9 || fail "Another camera setup is running"
if ((INSTALL_SYSTEM)); then
    command -v apt-get >/dev/null || fail "--system-deps requires apt-get"
    SUDO=()
    if ((EUID != 0)); then command -v sudo >/dev/null || fail "sudo is required"; SUDO=(sudo); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends python3-venv ffmpeg v4l-utils
fi
command -v "$PYTHON_BIN" >/dev/null || fail "Python not found: $PYTHON_BIN"
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3,10))'; then
    ((EXPLICIT_PYTHON == 0)) || fail "The supplied Python must be >= 3.10"
    BOOTSTRAP="$SCRIPT_DIR/.tools/bootstrap"
    mkdir -p -- "$SCRIPT_DIR/.tools"
    "$PYTHON_BIN" -m venv "$BOOTSTRAP" || fail "Install python3-venv or repeat with --system-deps"
    "$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: -r "$SCRIPT_DIR/requirements-bootstrap.txt"
    export UV_PYTHON_INSTALL_DIR="$SCRIPT_DIR/.tools/python"
    export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
    "$BOOTSTRAP/bin/uv" python install 3.12
    PYTHON_BIN=$("$BOOTSTRAP/bin/uv" python find --managed-python 3.12)
fi
command -v v4l2-ctl >/dev/null || fail "Install v4l-utils or repeat with --system-deps"
ENV_DIR="$SCRIPT_DIR/.venv"
[[ ! -L "$ENV_DIR" ]] || fail "Refusing a symlink environment: $ENV_DIR"
if [[ -e "$ENV_DIR" ]]; then
    [[ -f "$ENV_DIR/.setup-profile" && $(cat "$ENV_DIR/.setup-profile") == camera-v1 ]] || fail "Existing directory is not managed by this setup: $ENV_DIR"
else
    mkdir -p -- "$ENV_DIR"
    echo camera-v1 > "$ENV_DIR/.setup-profile"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR" || fail "venv creation failed; install python3-venv (or the matching version's venv package), then repeat"
fi
PY="$ENV_DIR/bin/python"
"$PY" -c 'import sys; assert sys.version_info >= (3,10), "Existing venv requires recreation with Python >= 3.10"'
REQ="$SCRIPT_DIR/requirements.txt"
if ((WITH_TESTS)); then REQ="$SCRIPT_DIR/requirements-dev.txt"; fi
"$PY" -m pip install --disable-pip-version-check --only-binary=:all: -r "$REQ"
"$PY" - "$ENV_DIR" <<'PY'
from pathlib import Path
import shutil, subprocess, sys
binary = shutil.which('ffmpeg')
if not binary:
    try:
        import imageio_ffmpeg
        binary = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise SystemExit('FFmpeg unavailable; run setup.sh --system-deps') from exc
binary = Path(binary).resolve()
destination = Path(sys.argv[1])/'bin/ffmpeg'
if destination.resolve() != binary:
    destination.unlink(missing_ok=True)
    destination.symlink_to(binary)
devices = subprocess.run([str(binary), '-hide_banner', '-devices'], capture_output=True, text=True, check=True)
if 'video4linux2' not in devices.stdout + devices.stderr:
    raise SystemExit('This FFmpeg has no V4L2 input; install Ubuntu/Debian ffmpeg')
encoders = subprocess.run([str(binary), '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
if 'libx264' not in encoders.stdout + encoders.stderr:
    raise SystemExit('This FFmpeg has no libx264; install Ubuntu/Debian ffmpeg')
print('FFmpeg with V4L2 and libx264:', binary)
PY
"$PY" -m pip check
mkdir -p -- "$SCRIPT_DIR/runtime"
"$PY" -m pip freeze > "$SCRIPT_DIR/runtime/requirements-resolved.txt"
echo "Ready. Activate: source '$ENV_DIR/bin/activate'"
echo "List cameras: '$PY' '$SCRIPT_DIR/camera_streamer.py' --list-devices"
