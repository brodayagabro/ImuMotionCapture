#!/usr/bin/env bash
# Server profiles deliberately keep CPU/GPU ONNX Runtime in separate venvs.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN=/usr/bin/python3
EXPLICIT_PYTHON=0
DEVICE=cpu
INSTALL_SYSTEM=0
WITH_TESTS=0
DOWNLOAD_MODELS=1

usage() {
    cat <<'EOF'
Usage: ./setup.sh [--device cpu|cuda] [--python /path/to/python3]
                  [--system-deps] [--with-tests] [--skip-models]

Creates server/.venv-cpu or server/.venv-cuda (Python >= 3.12), installs
pinned packages, local RTMlib/FreeMoCap checkouts and verified MediaMTX.
If system Python is too old, a private Python 3.12 is downloaded automatically.
Safe to repeat; does not start a service or open a camera.
--system-deps     Install python3-venv, git, ca-certificates through apt/sudo.
--with-tests      Install the test dependencies.
--skip-models     Defer model download/initialization to the first application run.
CUDA requires an existing NVIDIA driver compatible with CUDA 13.
EOF
}
fail() { echo "ERROR: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --python) (($# >= 2)) || fail "--python requires a value"; PYTHON_BIN=$2; EXPLICIT_PYTHON=1; shift 2 ;;
        --device) (($# >= 2)) || fail "--device requires a value"; DEVICE=$2; shift 2 ;;
        --system-deps) INSTALL_SYSTEM=1; shift ;;
        --with-tests) WITH_TESTS=1; shift ;;
        --skip-models) DOWNLOAD_MODELS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ "$DEVICE" == cpu || "$DEVICE" == cuda ]] || fail "--device must be cpu or cuda"
[[ $(uname -s) == Linux ]] || fail "This setup supports Linux"
command -v flock >/dev/null || fail "flock (util-linux) is required"
mkdir -p -- "$SCRIPT_DIR/.tools"
exec 9>"$SCRIPT_DIR/.tools/setup.lock"
flock -n 9 || fail "Another server setup is running"
if ((INSTALL_SYSTEM)); then
    command -v apt-get >/dev/null || fail "--system-deps requires apt-get"
    SUDO=()
    if ((EUID != 0)); then command -v sudo >/dev/null || fail "sudo is required"; SUDO=(sudo); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends python3-venv git ca-certificates
fi
command -v "$PYTHON_BIN" >/dev/null || fail "Python not found: $PYTHON_BIN"
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3,12))'; then
    ((EXPLICIT_PYTHON == 0)) || fail "The supplied Python must be >= 3.12 for the pinned NumPy"
    BOOTSTRAP="$SCRIPT_DIR/.tools/bootstrap"
    mkdir -p -- "$SCRIPT_DIR/.tools"
    "$PYTHON_BIN" -m venv "$BOOTSTRAP" || fail "Install python3-venv or repeat with --system-deps"
    "$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: -r "$SCRIPT_DIR/requirements-bootstrap.txt"
    export UV_PYTHON_INSTALL_DIR="$SCRIPT_DIR/.tools/python"
    export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
    "$BOOTSTRAP/bin/uv" python install 3.12
    PYTHON_BIN=$("$BOOTSTRAP/bin/uv" python find --managed-python 3.12)
fi
command -v git >/dev/null || fail "Install git or repeat with --system-deps"
if [[ "$DEVICE" == cuda ]]; then
    command -v nvidia-smi >/dev/null || fail "NVIDIA driver is required for CUDA; use --device cpu on this machine"
    nvidia-smi -L
fi
ENV_DIR="$SCRIPT_DIR/.venv-$DEVICE"
[[ ! -L "$ENV_DIR" ]] || fail "Refusing a symlink environment: $ENV_DIR"
if [[ -e "$ENV_DIR" ]]; then
    [[ -f "$ENV_DIR/.setup-profile" && $(cat "$ENV_DIR/.setup-profile") == "server-$DEVICE-v1" ]] || fail "Existing directory is not managed by this setup: $ENV_DIR"
else
    mkdir -p -- "$ENV_DIR"
    echo "server-$DEVICE-v1" > "$ENV_DIR/.setup-profile"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR" || fail "venv creation failed; install python3-venv (or the matching version's venv package), then repeat"
fi
PY="$ENV_DIR/bin/python"
"$PY" -c 'import sys; assert sys.version_info >= (3,12), "Existing venv needs Python >= 3.12"'
"$PY" -m pip install --disable-pip-version-check --only-binary=:all: -r "$SCRIPT_DIR/requirements-$DEVICE.txt"
if ((WITH_TESTS)); then
    "$PY" -m pip install --disable-pip-version-check --only-binary=:all: -r "$SCRIPT_DIR/requirements-dev.txt"
fi
"$PY" - "$SCRIPT_DIR" "$ENV_DIR" "$DEVICE" <<'PY'
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, subprocess, sys, tarfile, tempfile, urllib.request
root, environment, device = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

# Never reset or overwrite an existing user's checkout.
repositories = root/'repositories'
repositories.mkdir(exist_ok=True)
git_env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], env=git_env, text=True).strip()
for line in (root/'requirements-repositories.txt').read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'):
        continue
    name, url, commit = line.split()
    if name not in ('rtmlib','freemocap') or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise SystemExit('Invalid pinned repository definition')
    path = repositories/name
    if path.exists():
        if git(path,'remote','get-url','origin')!=url or git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'):
            raise SystemExit(f'Existing checkout differs from pinned clean revision: {path}. Left unchanged.')
    else:
        with tempfile.TemporaryDirectory(prefix=f'.{name}-',dir=repositories) as staging:
            checkout = Path(staging)/'checkout'
            subprocess.run(['git','clone','--quiet','--filter=blob:none','--no-checkout',url,str(checkout)],env=git_env,check=True)
            subprocess.run(['git','-C',str(checkout),'checkout','--quiet','--detach',commit],env=git_env,check=True)
            checkout.rename(path)
    print(f'{name}: {commit}')

# Binary archives use committed SHA-256 values; no arbitrary archive paths
# are extracted. A valid cached archive is reused on subsequent runs.
spec = json.loads((root/'requirements-tools.json').read_text())['mediamtx']
arch = {'x86_64':'linux_amd64','aarch64':'linux_arm64','armv7l':'linux_armv7'}.get(platform.machine())
if arch not in spec['sha256']:
    raise SystemExit(f'Unsupported MediaMTX architecture: {platform.machine()}')
tools = root/'.tools'
(tools/'downloads').mkdir(parents=True,exist_ok=True)
(tools/'bin').mkdir(exist_ok=True)
name = f'mediamtx_v{spec["version"]}_{arch}.tar.gz'
archive = tools/'downloads'/name
expected = spec['sha256'][arch]
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
if not archive.exists() or sha(archive)!=expected:
    temporary = archive.with_suffix('.download')
    try:
        with urllib.request.urlopen(spec['base_url']+name,timeout=60) as response, temporary.open('wb') as output:
            import shutil
            shutil.copyfileobj(response,output)
        if sha(temporary)!=expected:
            raise SystemExit('MediaMTX checksum mismatch')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
with tarfile.open(archive) as bundle:
    member = bundle.getmember('mediamtx')
    if not member.isfile():
        raise SystemExit('MediaMTX archive member is not a regular file')
    temporary = tools/'bin/mediamtx.new'
    temporary.write_bytes(bundle.extractfile(member).read())
    temporary.chmod(0o755)
    temporary.replace(tools/'bin/mediamtx')
link = environment/'bin/mediamtx'
link.unlink(missing_ok=True)
link.symlink_to(tools/'bin/mediamtx')
subprocess.run([str(link),'--version'],check=True)

import cv2, numpy, fastapi, onnxruntime as ort
if not hasattr(cv2,'aruco'):
    raise SystemExit('OpenCV contrib/ArUco is missing')
wrong = 'onnxruntime' if device=='cuda' else 'onnxruntime-gpu'
try:
    importlib.metadata.version(wrong)
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(f'Conflicting {wrong} distribution in this environment')
if device=='cuda':
    ort.preload_dlls(directory='')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise SystemExit('CUDAExecutionProvider unavailable')
sys.path.insert(0,str(root))
import visual_mocap_server as server
server.load_triangulator(repositories)
server.load_bvh_module()
sys.path.insert(0,str(repositories/'rtmlib'))
import rtmlib
print('Server imports OK; ONNX providers:',ort.get_available_providers())
PY
if ((DOWNLOAD_MODELS)); then
    "$PY" - "$SCRIPT_DIR" "$DEVICE" <<'PY'
from pathlib import Path
import shutil, sys
root, device = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0,str(root))
import visual_mocap_server as server
# Reuse only the two known already-downloaded checkpoints when migrating
# this workstation. Otherwise RTMlib downloads them from its configured URLs.
old_cache = Path.home()/'.cache/rtmlib/hub/checkpoints'
new_cache = root/'.cache/rtmlib/hub/checkpoints'
new_cache.mkdir(parents=True,exist_ok=True)
for name in ['yolox_m_8xb8-300e_humanart-c2c7a14a.onnx',
             'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx']:
    if (old_cache/name).is_file() and not (new_cache/name).exists():
        shutil.copy2(old_cache/name,new_cache/name)
model=server.PoseModel(server.arguments(['--device',device]))
print('Balanced model sessions initialized:',model.providers)
PY
fi
"$PY" -m pip check
mkdir -p -- "$SCRIPT_DIR/runtime"
"$PY" -m pip freeze > "$SCRIPT_DIR/runtime/requirements-resolved-$DEVICE.txt"
echo "Ready. Activate: source '$ENV_DIR/bin/activate'"
echo "RTSP relay: '$PY' '$SCRIPT_DIR/rtsp_relay.py'"
echo "Demo: '$PY' '$SCRIPT_DIR/visual_mocap_server.py' --demo --device '$DEVICE'"
