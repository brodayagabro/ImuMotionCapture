#!/usr/bin/env python3
"""Run the local MediaMTX RTSP receiver. Install with ./setup.sh first."""
import argparse
import ipaddress
import json
import logging
import math
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0", type=ipaddress.ip_address)
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--mediamtx", default=str(HERE / ".tools/bin/mediamtx"))
    parser.add_argument("--duration", type=float, help="Optional wall-clock limit for testing")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    return args


def relay_config(bind, port):
    bind = str(bind)
    address = f"[{bind}]:{port}" if ":" in bind else f"{bind}:{port}"
    return ("logLevel: info\nrtsp: yes\nrtspTransports: [tcp]\nrtspAddress: " + json.dumps(address) + "\n"
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\napi: no\nmetrics: no\npprof: no\nplayback: no\n"
            "pathDefaults:\n  source: publisher\n  overridePublisher: false\npaths:\n  all_others:\n")


def run(args, stop):
    binary = shutil.which(args.mediamtx)
    if not binary:
        raise ValueError("MediaMTX missing; run ./setup.sh or specify --mediamtx /path/to/mediamtx")
    binary = str(Path(binary).resolve())
    with tempfile.TemporaryDirectory(prefix="visual-mocap-relay-") as directory:
        config = Path(directory) / "mediamtx.yml"
        config.write_text(relay_config(args.bind, args.port))
        process = subprocess.Popen([binary, str(config)], cwd=directory, start_new_session=True)
        try:
            deadline = time.monotonic() + args.duration if args.duration else math.inf
            logging.info("RTSP on %s:%s; rtsp://SERVER:%s/CAMERA_NAME", args.bind, args.port, args.port)
            while process.poll() is None and not stop.is_set() and time.monotonic() < deadline:
                stop.wait(.2)
            if process.poll() is not None:
                return process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
    return 0


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    try:
        return run(args, stop)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
