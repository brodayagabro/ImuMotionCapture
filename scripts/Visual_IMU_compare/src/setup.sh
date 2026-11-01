#!/usr/bin/env bash
# Desktop experiment host: no CUDA, FFmpeg, Noitom SDK or external checkouts.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN=/usr/bin/python3
WITH_TESTS=0
while (($#)); do
    case "$1" in
        --python) (($# >= 2)) || { echo '--python needs a value' >&2; exit 1; }; PYTHON_BIN=$2; shift 2 ;;
        --with-tests) WITH_TESTS=1; shift ;;
        -h|--help) echo 'Usage: ./setup.sh [--python /path/to/python3] [--with-tests]'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Use Python >= 3.11"'
mkdir -p -- "$SCRIPT_DIR/.tools"
exec 9>"$SCRIPT_DIR/.tools/setup.lock"
flock -n 9 || { echo 'Another experiment setup is running' >&2; exit 1; }
ENV_DIR="$SCRIPT_DIR/.venv"
[[ ! -L "$ENV_DIR" ]] || { echo 'Refusing a symlink environment' >&2; exit 1; }
if [[ -e "$ENV_DIR" ]]; then
    [[ -f "$ENV_DIR/.setup-profile" && $(cat "$ENV_DIR/.setup-profile") == visual-experiment-v1 ]] || {
        echo "Existing unmanaged environment: $ENV_DIR" >&2; exit 1;
    }
else
    mkdir -p -- "$ENV_DIR"
    echo visual-experiment-v1 > "$ENV_DIR/.setup-profile"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR" || { echo 'Install python3-venv, then repeat setup' >&2; exit 1; }
fi
REQ="$SCRIPT_DIR/requirements.txt"
if ((WITH_TESTS)); then REQ="$SCRIPT_DIR/requirements-dev.txt"; fi
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: -r "$REQ"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -c 'import numpy, OpenGL, pyqtgraph; from PyQt6 import QtWidgets; print("Experiment imports OK")'
mkdir -p -- "$SCRIPT_DIR/runtime"
"$ENV_DIR/bin/python" -m pip freeze > "$SCRIPT_DIR/runtime/requirements-resolved.txt"
echo "Ready: '$ENV_DIR/bin/python' '$SCRIPT_DIR/main.py' --source-ip 192.168.1.90"
