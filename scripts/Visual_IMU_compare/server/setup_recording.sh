#!/usr/bin/env bash
# Shared installer called by camera/setup.sh and server/setup.sh --workflow recording.
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROLE=${1:?Expected camera or server}; shift
[[ "$ROLE" == camera || "$ROLE" == server ]] || exit 2
ROOT="$(cd -- "$HERE/../$ROLE" && pwd -P)"
PYTHON_BIN=/usr/bin/python3
SYSTEM_DEPS=0
WITH_TESTS=0
while (($#)); do
    case "$1" in
        --python) PYTHON_BIN=${2:?Python path required}; shift 2 ;;
        --system-deps) SYSTEM_DEPS=1; shift ;;
        --with-tests) WITH_TESTS=1; shift ;;
        -h|--help)
            echo "Usage: $ROLE/setup.sh --workflow recording [--python python3] [--system-deps] [--with-tests]"
            echo "Installs a private Python 3.12 and the pinned native SkellyCam/FreeMoCap environment."
            echo "No sudo unless --system-deps is explicitly supplied. Does not start services."
            exit 0 ;;
        *) echo "Unknown recording-profile argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$ROOT/.tools" "$ROOT/.cache" "$ROOT/runtime"
exec 9>"$ROOT/.tools/recording-setup.lock"
flock -n 9 || { echo 'Recording setup already running' >&2; exit 1; }
if ((SYSTEM_DEPS)); then
    SUDO=(); if ((EUID != 0)); then SUDO=(sudo); fi
    PACKAGES=(python3-venv git ca-certificates ffmpeg libgl1 libglib2.0-0)
    if [[ "$ROLE" == camera ]]; then PACKAGES+=(libportaudio2 v4l-utils); else PACKAGES+=(blender); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends "${PACKAGES[@]}"
fi
for TOOL in git ffmpeg ffprobe; do
    command -v "$TOOL" >/dev/null || { echo "Missing $TOOL: install it or repeat with --system-deps" >&2; exit 1; }
done
BOOTSTRAP="$ROOT/.tools/recording-bootstrap"
if [[ ! -x "$BOOTSTRAP/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$BOOTSTRAP" || { echo 'Install python3-venv or repeat with --system-deps' >&2; exit 1; }
fi
"$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements-bootstrap.txt"
export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
export UV_CACHE_DIR="$ROOT/.cache/uv"
UV="$BOOTSTRAP/bin/uv"
"$UV" python install --no-bin 3.12
"$BOOTSTRAP/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import os, subprocess, sys
root = Path(sys.argv[1])
env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], env=env, text=True).strip()
for line in (root / 'requirements-recording-repositories.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    name, url, commit = line.split()
    target = root / 'repositories' / name
    if target.is_symlink(): raise SystemExit(f'Refusing symlink checkout: {target}')
    if target.exists():
        if git(target, 'remote', 'get-url', 'origin').removesuffix('.git') != url.removesuffix('.git'):
            raise SystemExit(f'Unexpected origin: {target}')
        if git(target, 'status', '--porcelain'):
            raise SystemExit(f'Local source modifications in {target}; preserve/commit them before setup')
        if git(target, 'rev-parse', 'HEAD') != commit:
            raise SystemExit(f'Checkout has a different commit: {target}; move it aside before setup')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(target)], env=env, check=True)
        git(target, 'checkout', '--detach', commit)
PY
if [[ "$ROLE" == camera ]]; then
    PROJECT="$ROOT/repositories/skellycam"
    ENV_DIR="$ROOT/.venv-skellycam"
    GROUP=()
else
    PROJECT="$ROOT/repositories/freemocap"
    ENV_DIR="$ROOT/.venv-freemocap"
    GROUP=(--group cpu)
fi
[[ ! -L "$ENV_DIR" ]] || { echo "Refusing symlink environment: $ENV_DIR" >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
# --locked rejects upstream lock drift; it never updates branches or dependencies.
"$UV" sync --project "$PROJECT" --python 3.12 --locked --no-default-groups "${GROUP[@]}"
if [[ "$ROLE" == server ]]; then
    "$UV" pip install --python "$ENV_DIR/bin/python" -r "$ROOT/requirements-recording.txt"
    "$BOOTSTRAP/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import shutil, sys
root = Path(sys.argv[1])
source = root / 'repositories/skellycam'
target = root / '.tools/skellycam-compatible'
if target.exists(): shutil.rmtree(target)
target.mkdir()
shutil.copytree(source / 'skellycam', target / 'skellycam', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('pyproject.toml', 'README.md', 'LICENSE'):
    shutil.copyfile(source / name, target / name)
project = (target / 'pyproject.toml').read_text()
old = '"opencv-contrib-python==4.12.*"'
assert project.count(old) == 1, 'Upstream OpenCV requirement changed; recheck compatibility fix'
(target / 'pyproject.toml').write_text(project.replace(old, '"opencv-contrib-python==4.14.*"'))
PY
    "$UV" pip install --python "$ENV_DIR/bin/python" --no-deps --reinstall "$ROOT/.tools/skellycam-compatible"
fi
if ((WITH_TESTS)); then "$UV" pip install --python "$ENV_DIR/bin/python" pytest==9.0.2; fi
"$UV" pip check --python "$ENV_DIR/bin/python"
export FREEMOCAP_BASE_FOLDER="$ROOT/runtime/native-check"
export SKELLYCAM_BASE_FOLDER="$ROOT/runtime/native-check"
"$ENV_DIR/bin/python" - "$ROLE" <<'PY'
import sys
from skellycam.core.ipc.process_management.worker_registry import WorkerRegistry
if sys.argv[1] == 'camera':
    try:
        from skellycam.core.camera_group.camera_group_manager import CameraGroupManager
    except OSError as exc:
        raise SystemExit(f'{exc}. Install libportaudio2 (or setup.sh --workflow recording --system-deps).')
    import av
    av.CodecContext.create('libx264', 'w')
else:
    from freemocap.core.pipeline.posthoc.posthoc_pipeline_manager import PosthocPipelineManager
    from skelly_synchronize.core.pipeline.runner import run_pipeline
print('Native recording APIs OK')
PY
"$UV" pip freeze --python "$ENV_DIR/bin/python" > "$ROOT/runtime/requirements-recording-resolved.txt"
echo "Ready: $ENV_DIR/bin/python"
if [[ "$ROLE" == server ]] && ! command -v blender >/dev/null; then
    echo 'Blender is required for BVH export. Install Blender or pass --blender /path/to/blender to freemocap_host.py.'
fi
