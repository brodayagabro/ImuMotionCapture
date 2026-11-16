#!/usr/bin/env bash
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON=/usr/bin/python3
SYSTEM=0 TESTS=0 CLEAN=0
fail() { echo "ERROR: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --python) (($# >= 2)) || fail "--python requires a path"; PYTHON=$2; shift 2 ;;
        --system-deps) SYSTEM=1; shift ;;
        --with-tests) TESTS=1; shift ;;
        --clean-legacy) CLEAN=1; shift ;;
        -h|--help) echo "Usage: ./setup.sh [--python /path/to/python3] [--system-deps] [--with-tests] [--clean-legacy]"; exit 0 ;;
        *) fail "Unknown argument: $1 (see README for migration from RTSP/SkellyCam)" ;;
    esac
done
mkdir -p "$DIR/.tools"
exec 9>"$DIR/.tools/setup.lock"
flock -n 9 || fail "Another setup is running"
if ((SYSTEM)); then
    SUDO=(); if ((EUID)); then SUDO=(sudo); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends python3-venv ffmpeg v4l-utils alsa-utils
fi
command -v "$PYTHON" >/dev/null || fail "Python not found: $PYTHON"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Camera needs Python >= 3.10"'
for tool in ffmpeg ffprobe v4l2-ctl arecord; do
    command -v "$tool" >/dev/null || fail "$tool missing. Install system dependencies (--system-deps), or add your binaries to PATH."
done
features=$(ffmpeg -hide_banner -h full 2>/dev/null)
[[ $features == *-isync* ]] || fail "Camera needs FFmpeg with -isync (5.1+). Install a newer FFmpeg; system Ubuntu 22.04 FFmpeg 4.4 is too old."
devices=$(ffmpeg -hide_banner -devices 2>&1)
[[ $devices == *video4linux2* && $devices == *alsa* ]] || fail "FFmpeg needs V4L2 and ALSA inputs"
ENV="$DIR/.venv"
[[ ! -L "$ENV" ]] || fail "Refusing a symlink environment"
# Never remove an active interpreter, recordings, or external environments.
"$PYTHON" - "$DIR" "$CLEAN" <<'PY_GUARD'
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        cmd=' '.join(argv)
        cwd=(proc/'cwd').resolve()
        if str(root) in cmd or cwd==root:
            if any(Path(arg).name in ('camera_streamer.py','skellycam_node.py','visual_mocap_server.py','freemocap_recording_server.py','rtsp_relay.py') for arg in argv):
                raise SystemExit(f'Stop the old service/application before setup (PID {proc.name})')
    except (OSError,PermissionError):pass
PY_GUARD
if [[ -d "$ENV" && $(cat "$ENV/.setup-profile" 2>/dev/null || true) != camera-local-v2 ]]; then
    [[ $(cat "$ENV/.setup-profile" 2>/dev/null || true) == camera-v1 ]] || fail "Unmanaged .venv: choose another directory or remove it manually"
    [[ ! -e "$DIR/.venv-before-local" ]] || fail "Previous migration backup exists"
    mv -- "$ENV" "$DIR/.venv-before-local"
    if ! "$PYTHON" -m venv --without-pip "$ENV"; then
        rm -rf -- "$ENV"; mv -- "$DIR/.venv-before-local" "$ENV"; fail "venv creation failed"
    fi
    echo camera-local-v2 > "$ENV/.setup-profile"
    rm -rf -- "$DIR/.venv-before-local"
else
    if [[ ! -d "$ENV" ]]; then
        "$PYTHON" -m venv --without-pip "$ENV"
        echo camera-local-v2 > "$ENV/.setup-profile"
    fi
fi
# Preserve the resolved FFmpeg location when launching the venv by absolute path.
"$ENV/bin/python" - "$ENV" <<'PY_TOOLS'
from pathlib import Path
import shutil,sys
for name in ('ffmpeg','ffprobe'):
    source=Path(shutil.which(name)).resolve()
    target=Path(sys.argv[1])/'bin'/name
    if target.resolve()!=source:
        target.unlink(missing_ok=True)
        target.symlink_to(source)
PY_TOOLS
if ((TESTS)); then
    "$ENV/bin/python" -m ensurepip --upgrade
    "$ENV/bin/python" -m pip install --disable-pip-version-check -r "$DIR/requirements-dev.txt"
fi
"$ENV/bin/python" "$DIR/camera_streamer.py" --help >/dev/null
if ((CLEAN)); then
    "$PYTHON" - "$DIR" <<'PY_CLEAN'
from pathlib import Path
import shutil,subprocess,sys
root=Path(sys.argv[1]).resolve()
# Explicit list of generated runtimes from previous versions. Capture data is excluded.
entries=['.tools/bootstrap', '.tools/python', '.tools/skellycam-compatible', '.cache/uv', '.venv-skellycam', '.tools/recording-bootstrap', '.tools/skellycam-compatible', '.cache/uv', 'repositories/skellycam']
for relative in dict.fromkeys(entries):
    path=root/relative
    if not path.exists():continue
    if path.is_symlink():print('Preserved symlink:',path);continue
    if (root/'.venv/bin/python').resolve().is_relative_to(path):
        print('Preserved Python used by the new environment:',path);continue
    if not path.resolve().is_relative_to(root):raise SystemExit(f'Outside project: {path}')
    if (path/'.git').exists():
        changed=subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True)
        if changed.strip():print('Preserved modified checkout:',path);continue
    in_use=False
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command=(proc/'cmdline').read_bytes().decode(errors='replace')
            if str(path) in command or (proc/'exe').resolve().is_relative_to(path):in_use=True
        except OSError:pass
    if in_use:raise SystemExit(f'Stop processes using {path} before cleanup')
    if path.is_dir():shutil.rmtree(path)
    else:path.unlink()
    print('Removed obsolete runtime:',path)
PY_CLEAN
fi
printf 'Ready: source "%s/bin/activate"
' "$ENV"
