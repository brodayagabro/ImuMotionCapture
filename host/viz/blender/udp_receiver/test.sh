#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_BIN="${BLENDER_BIN:-blender}"
TEST_TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf -- "$TEST_TMP_DIR"
}
trap cleanup EXIT

PYTHONPYCACHEPREFIX="$TEST_TMP_DIR/pycache" \
  python3 -m py_compile \
  "$SCRIPT_DIR/build_udp_blend.py" \
  "$SCRIPT_DIR/udp_mocap.py" \
  "$SCRIPT_DIR/test_udp_mocap.py" \
  "$SCRIPT_DIR/test_hardware.py"

"$BLENDER_BIN" --background --python-exit-code 1 \
  --python "$SCRIPT_DIR/build_udp_blend.py" -- \
  "$SCRIPT_DIR/Human_spine_N_sensors.blend1" \
  "$TEST_TMP_DIR/Human_spine_UDP.blend" \
  "$SCRIPT_DIR/udp_mocap.py"

"$BLENDER_BIN" --background "$TEST_TMP_DIR/Human_spine_UDP.blend" \
  --python-exit-code 1 \
  --python "$SCRIPT_DIR/test_udp_mocap.py" -- \
  "$SCRIPT_DIR/udp_mocap.py"
