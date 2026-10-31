#!/usr/bin/env python3
"""RTSP → pretrained RTMPose → FreeMoCap triangulation → WebGL.

One application file, including the browser UI. See README.md for deployment,
camera calibration, timing limitations and the JSON/WebSocket contract.
Python >= 3.11. Run ./setup.sh --device cpu or cuda to install dependencies.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import importlib.util
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("visual-mocap")
JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle"]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def default_repos():
    return HERE / "repositories"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", type=Path, default=default_repos())
    p.add_argument("--camera", action="append", default=[], metavar="NAME=RTSP_URL", help="Repeat for each camera; USB index or video path also accepted")
    p.add_argument("--config", type=Path, help='JSON: {"cameras": [{"id": "cam1", "source": "rtsp://...", "offset_ms": 0}]}')
    p.add_argument("--demo", action="store_true", help="Synthetic three-camera demo; no models or real cameras")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cuda:1, ... or cpu (explicit CPU fallback)")
    p.add_argument("--preset", choices=["lightweight", "balanced", "performance"], default="balanced")
    p.add_argument("--det-model", help="Optional matching local YOLOX ONNX file")
    p.add_argument("--pose-model", help="Optional matching 17-joint RTMPose ONNX file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fps", type=float, default=20, help="Maximum inference cycles per second, across all cameras")
    p.add_argument("--preview-width", type=int, default=640)
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--max-skew-ms", type=float, default=50, help="Allowed difference of corrected SERVER RECEPTION times, not exposure times")
    p.add_argument("--max-age-ms", type=float, default=500)
    p.add_argument("--max-reprojection-px", type=float, default=12)
    p.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
    p.add_argument("--calibration", type=Path, default=HERE / "runtime" / "calibration.json")
    p.add_argument("--board", default="7x5", help="ChArUco board square counts, columns x rows")
    p.add_argument("--square-m", type=float, default=0.04)
    p.add_argument("--marker-m", type=float, default=0.03)
    p.add_argument("--bvh-target", help="Destination HOST:PORT; visual-mocap-bvh/1 protocol")
    p.add_argument("--bvh-transport", choices=["udp", "tcp"], default="udp")
    p.add_argument("--bvh-file", type=Path, help="Also record a regular BVH file; finalized on Ctrl+C")
    p.add_argument("--bvh-units", choices=["m", "cm"], default="cm")
    p.add_argument("--bvh-warmup", type=int, default=20, help="Fully visible poses used to estimate fixed bone lengths")
    args = p.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        p.error("--device must be cpu or cuda[:GPU_INDEX]")
    for key in ("fps", "max_skew_ms", "max_age_ms", "max_reprojection_px", "square_m", "marker_m"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.preview_width < 160 or not 0 < args.confidence <= 1 or not 1 <= args.port <= 65535:
        p.error("Invalid preview width, confidence or port")
    if not re.fullmatch(r"\d+x\d+", args.board) or min(map(int, args.board.split("x"))) < 3:
        p.error("--board must be e.g. 7x5, at least 3x3 squares")
    if args.marker_m >= args.square_m:
        p.error("--marker-m must be smaller than --square-m")
    if args.demo and (args.camera or args.config):
        p.error("--demo cannot be mixed with real camera sources")
    if args.bvh_warmup < 1 or not 1 <= args.fps <= 100:
        p.error("--bvh-warmup must be positive; --fps must be between 1 and 100")
    return args


if __name__ == "__main__":
    CLI_ARGS = arguments()

try:
    import cv2
    import numpy as np
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Run ./setup.sh --device cpu (or cuda) in the server directory.") from exc


@dataclass(frozen=True)
class Source:
    id: str
    source: str | int
    offset_ms: float = 0.0


@dataclass(frozen=True)
class Frame:
    sequence: int
    received: float
    aligned: float
    image: np.ndarray
    synthetic_points: np.ndarray | None = None


def camera_sources(args):
    if args.demo:
        return [Source(f"cam{i+1}", "synthetic") for i in range(3)]
    entries = []
    if args.config:
        entries.extend(json.loads(args.config.read_text())["cameras"])
    for item in args.camera:
        name, separator, source = item.partition("=")
        if not separator:
            raise ValueError("Use --camera NAME=RTSP_URL")
        entries.append({"id": name, "source": source})
    result = []
    for entry in entries:
        name, source = entry["id"], entry["source"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name):
            raise ValueError("Camera IDs must contain 1–48 ASCII letters, digits, '_' or '-'")
        if not isinstance(source, (str, int)) or isinstance(source, bool) or source == "":
            raise ValueError(f"Invalid source for {name}")
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        offset = float(entry.get("offset_ms", 0))
        if not math.isfinite(offset):
            raise ValueError(f"Invalid timing offset for {name}")
        result.append(Source(name, source, offset))
    if not result or len({s.id for s in result}) != len(result) or len(result) > 8:
        raise ValueError("Supply 1–8 uniquely named cameras; 3D needs at least two calibrated cameras")
    return result


def load_triangulator(repos):
    path = repos / "freemocap/freemocap/core/tasks/triangulation/helpers/triangulate_simple.py"
    if not path.is_file():
        raise ValueError(f"Missing FreeMoCap triangulation helper: {path}")
    spec = importlib.util.spec_from_file_location("_freemocap_dlt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.triangulate_simple


def checked_calibration(data, ids):
    if data.get("schema_version") != 1 or data.get("units") != "m":
        raise ValueError("Calibration requires schema_version=1, units=m")
    if data.get("extrinsics_convention") != "world_to_camera":
        raise ValueError("Calibration extrinsics must be world_to_camera: Xc = R @ Xw + t")
    cameras = {}
    for name in ids:
        if name not in data.get("cameras", {}):
            raise ValueError(f"Calibration missing camera {name}")
        entry = data["cameras"][name]
        k = np.asarray(entry["K"], dtype=float)
        d = np.asarray(entry["D"], dtype=float).reshape(-1)
        r = np.asarray(entry["R"], dtype=float)
        t = np.asarray(entry["t"], dtype=float).reshape(-1)
        size = entry["image_size"]
        if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,) or d.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"Invalid matrix dimensions for {name}")
        if not all(np.isfinite(x).all() for x in (k, d, r, t)):
            raise ValueError(f"Nonfinite calibration for {name}")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError(f"Invalid intrinsics for {name}")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
            raise ValueError(f"Invalid rotation for {name}")
        if len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in size):
            raise ValueError(f"Invalid image_size for {name}")
        cameras[name] = dict(K=k, D=d, R=r, t=t, E=np.column_stack((r, t)), image_size=tuple(size))
    if len(cameras) < 2:
        raise ValueError("At least two calibrated cameras are required")
    centers = [-c["R"].T @ c["t"] for c in cameras.values()]
    if max(np.linalg.norm(a-b) for a, b in itertools.combinations(centers, 2)) < 1e-4:
        raise ValueError("Camera centers coincide; 3D cannot be triangulated")
    return cameras


def project(points, camera):
    pixels, _ = cv2.projectPoints(np.asarray(points, dtype=float), camera["R"], camera["t"], camera["K"], camera["D"])
    return pixels.reshape(-1, 2)


def reconstruct(observations, calibration, triangulate, confidence=0.4, max_error=12):
    """Triangulate only observed joints; reject poor geometry and reprojection.

    Pair hypotheses make a third camera useful for rejecting an outlier. This
    is single-person capture: detections with multiple people are excluded.
    """
    xyz, errors, scores = [None] * 17, [None] * 17, [0.0] * 17
    for joint in range(17):
        views = []
        for name, (points, likelihood) in observations.items():
            if name not in calibration or likelihood[joint] < confidence or not np.isfinite(points[joint]).all():
                continue
            c = calibration[name]
            normalized = cv2.undistortPoints(points[joint].reshape(1, 1, 2).astype(float), c["K"], c["D"]).reshape(2)
            ray = c["R"].T @ np.array([*normalized, 1.0])
            views.append((name, c, normalized, ray / np.linalg.norm(ray), float(likelihood[joint])))
        if len(views) < 2:
            continue

        def fit(indices):
            selected = [views[i] for i in indices]
            angles = [math.degrees(math.acos(float(np.clip(np.dot(a[3], b[3]), -1, 1))))
                      for a, b in itertools.combinations(selected, 2)]
            if not angles or max(min(a, 180-a) for a in angles) < 1.0:
                return None
            with np.errstate(divide="ignore", invalid="ignore"):
                point = triangulate(points=np.array([v[2] for v in selected]),
                                    extrinsics_mats=np.array([v[1]["E"] for v in selected]))
            if not np.isfinite(point).all():
                return None
            residuals = []
            for name, c, _, _, _ in views:
                camera_point = c["R"] @ point + c["t"]
                error = np.linalg.norm(project([point], c)[0] - observations[name][0][joint])
                residuals.append(float(error) if 0.05 < camera_point[2] < 50 else math.inf)
            return point, np.array(residuals)

        best = None
        candidates = [tuple(range(len(views))), *itertools.combinations(range(len(views)), 2)]
        for indices in candidates:
            fitted = fit(indices)
            if fitted is None:
                continue
            point, residuals = fitted
            inliers = np.flatnonzero(residuals <= max_error)
            if len(inliers) < 2:
                continue
            rank = (len(inliers), -float(np.mean(residuals[inliers])))
            if best is None or rank > best[0]:
                best = rank, inliers
        if best is None:
            continue
        fitted = fit(best[1])
        if fitted is None or np.any(fitted[1][best[1]] > max_error):
            continue
        xyz[joint] = fitted[0].tolist()
        errors[joint] = float(np.mean(fitted[1][best[1]]))
        scores[joint] = float(np.mean([views[i][4] for i in best[1]]))
    return dict(points=xyz, confidence=scores, reprojection_px=errors)


def select_frames(histories, now, max_age, max_skew):
    """Newest group with at least two views within a bounded time window.

    A stalled camera never blocks the others. No unbounded work queue exists.
    """
    fresh = {name: [f for f in frames if 0 <= now - f.received <= max_age]
             for name, frames in histories.items()}
    fresh = {name: frames for name, frames in fresh.items() if frames}
    best, best_key = {}, (0, -math.inf, 0)
    for anchor in sorted({frames[-1].aligned for frames in fresh.values()}, reverse=True):
        group = {}
        for name, frames in fresh.items():
            candidates = [f for f in frames if anchor - max_skew <= f.aligned <= anchor + 1e-9]
            if candidates:
                group[name] = max(candidates, key=lambda f: f.aligned)
        # Prefer a current stereo pair over an older three-camera group.
        key = min(len(group), 2), anchor, len(group)
        if key > best_key:
            best, best_key = group, key
    return best


def jpeg(image, width):
    if image.shape[1] > width:
        image = cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return data.tobytes()


def overlay(image, points, scores, threshold):
    image = image.copy()
    valid = np.isfinite(points).all(axis=1) & (scores >= threshold)
    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (80, 230, 120), 2, cv2.LINE_AA)
    for point in points[valid]:
        cv2.circle(image, tuple(point.astype(int)), 4, (40, 180, 255), -1, cv2.LINE_AA)
    return image


def demo_calibration(ids):
    cameras = {}
    for i, name in enumerate(ids):
        angle = math.radians((i - 1) * 35)
        r, _ = cv2.Rodrigues(np.array([0, angle, 0], dtype=float))
        cameras[name] = dict(K=[[700, 0, 400], [0, 700, 380], [0, 0, 1]], D=[0]*5,
                             R=r.tolist(), t=[0, 0, 4], image_size=[800, 600])
    return dict(schema_version=1, units="m", extrinsics_convention="world_to_camera",
                world_frame="synthetic: X right, Y down, Z forward", cameras=cameras)


def demo_points(t):
    points = np.array([[0,-1.7,0],[-.035,-1.74,-.02],[.035,-1.74,-.02],[-.08,-1.7,0],[.08,-1.7,0],
                       [-.22,-1.45,0],[.22,-1.45,0],[-.38,-1.15,0],[.38,-1.15,0],
                       [-.5,-.9,0],[.5,-.9,0],[-.16,-.85,0],[.16,-.85,0],
                       [-.16,-.45,0],[.16,-.45,0],[-.16,0,0],[.16,0,0]], dtype=float)
    points[:, 0] += .2 * math.sin(t * .6)
    points[9, 1] -= .55 * (1 + math.sin(t * 1.5))
    points[7, 1] -= .25 * (1 + math.sin(t * 1.5))
    points[10, 2] += .3 * math.sin(t)
    return points


class Camera:
    def __init__(self, source, stop, width, demo_camera=None):
        self.source, self.stop, self.width = source, stop, width
        self.demo_camera = demo_camera
        self.lock = threading.Lock()
        self.frames = deque(maxlen=8)
        self.preview = None
        self.status = "connecting"
        self.sequence = 0
        self.fps = 0.0
        self.last_jpeg = 0.0
        self.thread = threading.Thread(target=self.run, name=f"camera-{source.id}", daemon=True)

    def history(self):
        with self.lock:
            return list(self.frames)

    def stats(self, now):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            age = (now - frame.received) * 1000 if frame else None
            return dict(id=self.source.id, status=self.status, fps=round(self.fps, 1),
                        age_ms=round(age, 1) if age is not None else None,
                        sequence=frame.sequence if frame else None,
                        size=[frame.image.shape[1], frame.image.shape[0]] if frame else None)

    def publish(self, image, synthetic=None):
        now = time.monotonic()
        self.sequence += 1
        frame = Frame(self.sequence, now, now + self.source.offset_ms / 1000, image, synthetic)
        encoded = jpeg(image, self.width) if now - self.last_jpeg >= 1/15 else None
        with self.lock:
            if self.frames:
                interval = now - self.frames[-1].received
                self.fps = .9 * self.fps + .1 / max(interval, .001)
            self.frames.append(frame)
            self.status = "online"
            if encoded:
                self.preview = (self.sequence, now, encoded)
                self.last_jpeg = now

    def run(self):
        if self.demo_camera:
            while not self.stop.is_set():
                started = time.monotonic()
                points = project(demo_points(started), self.demo_camera)
                image = np.full((600, 800, 3), (32, 27, 22), dtype=np.uint8)
                image = overlay(image, points, np.ones(17), .4)
                cv2.putText(image, "SYNTHETIC DEMO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (180, 190, 200), 2)
                self.publish(image, points)
                self.stop.wait(max(0, 1/30 - (time.monotonic() - started)))
            return
        while not self.stop.is_set():
            cap = None
            try:
                source = self.source.source
                is_file = isinstance(source, str) and "://" not in source and Path(source).is_file()
                if isinstance(source, int):
                    cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
                if not cap.isOpened():
                    raise OSError("Could not open stream")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                file_fps = cap.get(cv2.CAP_PROP_FPS)
                file_fps = file_fps if math.isfinite(file_fps) and 0 < file_fps < 500 else 30
                deadline = time.monotonic()
                while not self.stop.is_set():
                    ok, image = cap.read()
                    if not ok or image is None:
                        break
                    self.publish(image)
                    if is_file:
                        deadline += 1/file_fps
                        self.stop.wait(max(0, deadline-time.monotonic()))
            except Exception:
                # Do not echo RTSP URLs/passwords into the browser or our logs.
                LOG.warning("Camera %s disconnected; retrying", self.source.id)
            finally:
                if cap is not None:
                    cap.release()
                with self.lock:
                    self.status = "reconnecting"
                    self.frames.clear()
                    self.preview = None
            self.stop.wait(1)


class PoseModel:
    def __init__(self, args):
        root = (args.repos / "rtmlib").resolve()
        if not (root / "rtmlib/__init__.py").is_file():
            raise ValueError(f"Missing RTMlib checkout: {root}")
        sys.path.insert(0, str(root))
        os.environ.setdefault("TORCH_HOME", str(HERE / ".cache" / "rtmlib"))
        import onnxruntime as ort
        if args.device.startswith("cuda"):
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider unavailable. Use ./setup.sh --device cuda and a supported NVIDIA driver; --device cpu is explicit CPU mode.")
        from rtmlib import Body
        config = Body.MODE[args.preset]
        self.body = Body(mode=args.preset, backend="onnxruntime", device=args.device,
                         det=args.det_model, pose=args.pose_model,
                         det_input_size=config["det_input_size"], pose_input_size=config["pose_input_size"])
        self.providers = {}
        for name, tool in (("detector", self.body.det_model), ("pose", self.body.pose_model)):
            providers = tool.session.get_providers()
            if args.device.startswith("cuda") and "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"{name}: CUDA initialization failed; refusing silent CPU fallback")
            tool.session.disable_fallback()
            self.providers[name] = providers

    def infer(self, frame):
        boxes = self.body.det_model(frame.image)
        # RTMPose itself treats [] as a full-image box. Do not hallucinate a
        # person when the detector found nobody; do not merge different people.
        if len(boxes) != 1:
            return np.zeros((17, 2)), np.zeros(17), len(boxes)
        points, scores = self.body.pose_model(frame.image, bboxes=boxes)
        if points.shape != (1, 17, 2) or scores.shape != (1, 17):
            raise ValueError("Expected a COCO-17 RTMPose model")
        scores = np.where(np.isfinite(scores[0]) & np.isfinite(points[0]).all(axis=1), scores[0], 0)
        return np.nan_to_num(points[0]), scores, 1


def load_bvh_module():
    path = next((parent / "host/storage/bvh.py" for parent in HERE.parents
                 if (parent / "host/storage/bvh.py").is_file()), None)
    if path is None:
        raise ValueError("Missing ImuMotionCapture/host/storage/bvh.py; keep the server inside the full repository")
    name = "_visual_mocap_host_bvh"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rotation_between(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        raise ValueError("Degenerate bone direction")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross = np.cross(a, b)
    cosine = float(np.clip(a @ b, -1, 1))
    if cosine < -1 + 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / max(1 + cosine, 1e-8)


def body_basis(right, up):
    x = right / max(np.linalg.norm(right), 1e-12)
    y = up - x * (x @ up)
    if min(np.linalg.norm(right), np.linalg.norm(y)) < 1e-6:
        raise ValueError("Degenerate torso geometry")
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def bvh_hierarchy(recording, module):
    lines, order = ["HIERARCHY"], []

    def vector(values):
        return " ".join(f"{v:.8f}" for v in recording.units_per_meter * module.TO_BVH @ values)

    def emit(index, depth):
        joint = recording.joints[index]
        pad = "  " * depth
        order.append(index)
        lines.extend([f"{pad}{'ROOT' if index == 0 else 'JOINT'} {joint.name}", pad + "{",
                      pad + "  OFFSET " + vector(joint.offset),
                      pad + "  CHANNELS " + ("6 Xposition Yposition Zposition " if index == 0 else "3 ") + "Zrotation Xrotation Yrotation"])
        children = [i for i, j in enumerate(recording.joints) if j.parent == index]
        for child in children:
            emit(child, depth + 1)
        if not children:
            lines.extend([pad + "  End Site", pad + "  {", pad + "    OFFSET " + vector(joint.end), pad + "  }"])
        lines.append(pad + "}")

    emit(0, 0)
    return "\n".join(lines) + "\n", order


class BvhFile:
    """Bounded-memory BVH writer, regular time grid with last-pose hold."""
    def __init__(self, path, hierarchy, fps):
        self.path, self.fps = path, fps
        if path.exists():
            raise ValueError(f"BVH output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(path.name + ".part")
        self.file = self.temporary.open("x", encoding="ascii", newline="\n")
        self.file.write(hierarchy + "MOTION\nFrames: ")
        self.count_position = self.file.tell()
        self.file.write(f"{0:012d}\nFrame Time: {1/fps:.12f}\n")
        self.count = 0
        self.origin = None
        self.previous = None
        self.last_flush = time.monotonic()

    def append(self, timestamp, channels):
        if self.origin is None:
            self.origin = timestamp
            self.previous = channels
        # No gap is silently compressed: missing measurements are held in
        # the file. The network frames retain their original timestamps.
        while self.origin + self.count / self.fps <= timestamp + 1e-8:
            values = channels if abs(self.origin + self.count/self.fps - timestamp) < 1e-8 else self.previous
            self.file.write(" ".join(f"{v:.8f}" for v in values) + "\n")
            self.count += 1
        self.previous = channels
        if time.monotonic() - self.last_flush > 1:
            self.file.flush()
            self.last_flush = time.monotonic()

    def close(self):
        self.file.seek(self.count_position)
        self.file.write(f"{self.count:012d}")
        self.file.flush()
        self.file.close()
        self.temporary.replace(self.path)


def bvh_rig(measurements):
    """Fixed rest offsets, in BVH metres; shared by export and the idle armature."""
    torso, shoulders, hips, head, la, lf, ra, rf, lt, ls, rt, rs = measurements
    specs = [
        ("Hips", None, (0,0,0)), ("Spine", 0, (0,0,0)), ("Chest", 1, (0,torso,0)),
        ("Head", 2, (0,head,0)), ("LeftArm", 2, (-shoulders/2,0,0)),
        ("LeftForeArm", 4, (-la,0,0)), ("LeftHand", 5, (-lf,0,0)),
        ("RightArm", 2, (shoulders/2,0,0)), ("RightForeArm", 7, (ra,0,0)),
        ("RightHand", 8, (rf,0,0)), ("LeftUpLeg", 0, (-hips/2,0,0)),
        ("LeftLeg", 10, (0,-lt,0)), ("LeftFoot", 11, (0,-ls,0)),
        ("RightUpLeg", 0, (hips/2,0,0)), ("RightLeg", 13, (0,-rt,0)),
        ("RightFoot", 14, (0,-rs,0)),
    ]
    return [dict(name=name, parent=parent, offset=offset,
                 end=(0,.1,0) if name == "Head" else (-.1,0,0) if name == "LeftHand"
                 else (.1,0,0) if name == "RightHand" else (0,0,.12) if name.endswith("Foot") else None)
            for name, parent, offset in specs]


def armature_geometry(rig, channels, tracked):
    """Forward kinematics of the exported ZXY channels, including End Sites.

    Channels and offsets are in metres here. Bone frames keep octahedron
    cross-sections attached to the parent rotation, as in human_canvas_gl.py.
    """
    positions, rotations, bones = [], [], []
    for i, joint in enumerate(rig):
        z, x, y = np.radians(channels[3+3*i:6+3*i])
        cz, sz, cx, sx, cy, sy = np.cos(z), np.sin(z), np.cos(x), np.sin(x), np.cos(y), np.sin(y)
        local = (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
                 np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]) @
                 np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]))
        parent = joint["parent"]
        if parent is None:
            position, rotation = np.asarray(channels[:3]) + joint["offset"], local
        else:
            position = positions[parent] + rotations[parent] @ joint["offset"]
            rotation = rotations[parent] @ local
            if np.linalg.norm(joint["offset"]) > 1e-8:
                bones.append(dict(name=rig[parent]["name"]+":"+joint["name"],
                                  start=positions[parent].tolist(), end=position.tolist(),
                                  axes=rotations[parent].T.tolist(), tracked=bool(tracked[parent])))
        positions.append(position)
        rotations.append(rotation)
        if joint["end"] is not None:
            bones.append(dict(name=joint["name"]+":End", start=position.tolist(),
                              end=(position + rotation @ joint["end"]).tolist(),
                              axes=rotation.T.tolist(), tracked=bool(tracked[i])))
    return dict(joint_names=[j["name"] for j in rig], points=[p.tolist() for p in positions], bones=bones)


class ArmatureView:
    """Display-only state: T-pose, three fresh BVH frames, then last-pose hold.

    Never generates or changes outbound BVH. All methods use Engine.bvh_lock.
    """
    def __init__(self, max_age_s):
        self.max_age_s = max_age_s
        self.identity = None
        self.rig = bvh_rig((.5,.4,.28,.22,.28,.25,.28,.25,.42,.42,.42,.42))
        self.geometry = self.rest_geometry()
        self.last_timestamp = None
        self.last_frame = -1
        self.display_timestamp = None
        self.display_frame = None
        self.consecutive = 0
        self.live = False

    def rest_geometry(self):
        channels = np.zeros(3+3*len(self.rig))
        geometry = armature_geometry(self.rig, channels, [False]*len(self.rig))
        channels[1] = -min(p[1] for p in geometry["points"])
        return armature_geometry(self.rig, channels, [False]*len(self.rig))

    def update(self, adapter, frame, now):
        self.live = False
        if frame is None:
            self.consecutive = 0
            return
        identity = (adapter.metadata["stream_id"], adapter.metadata["skeleton_id"])
        if identity != self.identity:
            self.__init__(self.max_age_s)
            self.identity = identity
            order = adapter.order
            index = {original: i for i, original in enumerate(order)}
            self.rig = [dict(name=j.name, parent=index.get(j.parent),
                             offset=adapter.module.TO_BVH @ np.array(j.offset),
                             end=adapter.module.TO_BVH @ np.array(j.end) if np.linalg.norm(j.end) else None)
                        for j in (adapter.recording.joints[i] for i in order)]
            self.geometry = self.rest_geometry()
        timestamp, frame_id = frame["timestamp_monotonic_s"], frame["frame_id"]
        channels = np.asarray(frame["channels"], dtype=float)
        if (not np.isfinite(timestamp) or not 0 <= now-timestamp <= self.max_age_s
                or channels.shape != (3+3*len(self.rig),) or not np.isfinite(channels).all()
                or len(frame["tracked"]) != len(self.rig)
                or frame_id <= self.last_frame
                or (self.last_timestamp is not None and timestamp <= self.last_timestamp)):
            self.consecutive = 0
            return
        if self.last_timestamp is not None and timestamp-self.last_timestamp > self.max_age_s:
            self.consecutive = 0
        self.last_timestamp, self.last_frame = timestamp, frame_id
        self.consecutive += 1
        if self.consecutive < 3:
            return
        if adapter.metadata["units"] == "cm":
            channels[:3] /= 100
        self.geometry = armature_geometry(self.rig, channels, frame["tracked"])
        self.display_timestamp, self.display_frame = timestamp, frame_id
        self.live = True

    def snapshot(self, now):
        age = now-self.display_timestamp if self.display_timestamp is not None else None
        live = self.live and age is not None and age <= self.max_age_s
        status = "live" if live else "held" if age is not None else "waiting" if self.identity else "rest"
        return dict(**self.geometry, status=status, units="m", rotation_order="ZXY",
                    rig_id=":".join(self.identity) if self.identity else "preview",
                    frame_id=self.display_frame, age_ms=age*1000 if age is not None else None,
                    consecutive_frames=self.consecutive, required_frames=3, max_age_ms=self.max_age_s*1000)


class BvhAdapter:
    """COCO-17 positions → fixed skeleton → host BVH channel convention.

    Swing-only limb fitting is deliberate: axial twist, wrist and foot
    orientations cannot be measured from these landmarks. Missing limbs
    hold their previous LOCAL rotation and are explicitly marked untracked.
    """
    def __init__(self, args):
        self.args, self.module = args, load_bvh_module()
        self.samples = []
        self.recording = None
        self.metadata = None
        self.writer = None
        self.file_error = None
        self.previous_local = {}
        self.stream_id = str(uuid.uuid4())
        self.status = "waiting_for_3d"
        self.axis = np.diag([1., -1., -1.])  # camera-world → BVH (Y up)

    def initialize(self, measurements):
        t = self.module.TO_BVH
        joints = []
        for joint in bvh_rig(measurements):
            name = joint["name"]
            end = joint["end"] or (0,0,0)
            joints.append(self.module.Joint(name, joint["parent"], tuple(t.T @ joint["offset"]), name, tuple(t.T @ end)))
        self.recording = self.module.BvhRecording(joints, [0,0,0], fps=self.args.fps,
                                                  units_per_meter=100 if self.args.bvh_units == "cm" else 1)
        hierarchy, self.order = bvh_hierarchy(self.recording, self.module)
        channel_names = ["Hips.Xposition", "Hips.Yposition", "Hips.Zposition"]
        channel_names += [f"{joints[i].name}.{axis}rotation" for i in self.order for axis in "ZXY"]
        self.metadata = dict(protocol="visual-mocap-bvh/1", type="skeleton", stream_id=self.stream_id,
                             skeleton_id=hashlib.sha256(hierarchy.encode()).hexdigest()[:16], hierarchy=hierarchy,
                             channel_names=channel_names, joint_names=[joints[i].name for i in self.order],
                             units=self.args.bvh_units, nominal_fps=self.args.fps, rotation_order="ZXY",
                             axes="X right, Y up, Z backward relative to calibration world",
                             orientation_method="torso basis + limb shortest-arc swing; axial twist unmeasured",
                             unmeasured_joints=["LeftHand", "RightHand", "LeftFoot", "RightFoot"])
        if self.args.bvh_file:
            try:
                self.writer = BvhFile(self.args.bvh_file, hierarchy, self.args.fps)
            except (OSError, ValueError) as exc:
                self.file_error = str(exc)

    def update(self, skeleton, timestamp, frame_id):
        if self.recording and self.recording.times and timestamp <= self.recording.times[-1]:
            self.status = "waiting_for_new_timestamp"
            return None
        points = [self.axis @ p if p is not None else None for p in skeleton["points"]]
        if any(points[i] is None for i in (5,6,11,12)):
            self.status = "torso_not_tracked"
            return None
        hips = (points[11]+points[12])/2
        chest = (points[5]+points[6])/2
        try:
            hip_basis = body_basis(points[12]-points[11], chest-hips)
            chest_basis = body_basis(points[6]-points[5], chest-hips)
        except ValueError:
            self.status = "degenerate_torso"
            return None
        if self.recording is None:
            needed = [0,5,6,7,8,9,10,11,12,13,14,15,16]
            if any(points[i] is None for i in needed):
                self.status = "warmup_requires_full_body"
                return None
            lengths = [np.linalg.norm(chest-hips), np.linalg.norm(points[6]-points[5]),
                       np.linalg.norm(points[12]-points[11]), np.linalg.norm(points[0]-chest)]
            lengths += [np.linalg.norm(points[a]-points[b]) for a,b in ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16))]
            if min(lengths) < .05 or max(lengths) > 1.2:
                self.status = "implausible_body_dimensions"
                return None
            self.samples.append(lengths)
            if len(self.samples) < self.args.bvh_warmup:
                self.status = f"warmup_{len(self.samples)}_of_{self.args.bvh_warmup}"
                return None
            self.initialize(np.median(self.samples, axis=0))
            self.samples.clear()
        world = {"Hips": hip_basis, "Spine": chest_basis, "Chest": chest_basis}
        tracked = {"Hips": True, "Spine": True, "Chest": True}

        def fit(name, parent, rest, start, end):
            valid = start is not None and end is not None and np.linalg.norm(end-start) > .02
            rotation = rotation_between(world[parent] @ rest, end-start) @ world[parent] if valid else world[parent] @ self.previous_local.get(name, np.eye(3))
            world[name], tracked[name] = rotation, bool(valid)

        fit("Head", "Chest", np.array([0.,1,0]), chest, points[0])
        for side, sign, shoulder, elbow, wrist, hip, knee, ankle in (
            ("Left",-1,5,7,9,11,13,15), ("Right",1,6,8,10,12,14,16)):
            fit(side+"Arm", "Chest", np.array([sign,0.,0]), points[shoulder], points[elbow])
            fit(side+"ForeArm", side+"Arm", np.array([sign,0.,0]), points[elbow], points[wrist])
            world[side+"Hand"] = world[side+"ForeArm"]
            tracked[side+"Hand"] = False
            fit(side+"UpLeg", "Hips", np.array([0.,-1,0]), points[hip], points[knee])
            fit(side+"Leg", side+"UpLeg", np.array([0.,-1,0]), points[knee], points[ankle])
            world[side+"Foot"] = world[side+"Leg"]
            tracked[side+"Foot"] = False
        for joint in self.recording.joints:
            parent = np.eye(3) if joint.parent is None else world[self.recording.joints[joint.parent].name]
            self.previous_local[joint.name] = parent.T @ world[joint.name]
        t = self.module.TO_BVH
        self.recording.append(timestamp, {name:t.T @ rotation @ t for name,rotation in world.items()}, root_position=t.T @ hips)
        raw = self.recording.frames[-1]
        channels = list(raw[:3]) + [v for i in self.order for v in raw[3+i*3:6+i*3]]
        # Keep only continuity state. Recording to disk is incremental.
        self.recording.frames[:] = self.recording.frames[-1:]
        self.recording.times[:] = self.recording.times[-1:]
        if self.writer:
            try:
                self.writer.append(timestamp, channels)
            except OSError as exc:
                self.file_error = str(exc)
                self.writer.file.close()
                self.writer = None
        self.status = "streaming_pose"
        return dict(protocol="visual-mocap-bvh/1", type="frame", stream_id=self.stream_id,
                    skeleton_id=self.metadata["skeleton_id"], frame_id=frame_id,
                    timestamp_monotonic_s=timestamp, channels=[round(float(v), 6) for v in channels],
                    tracked=[tracked[self.recording.joints[i].name] for i in self.order])

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def parse_target(target):
    host, separator, port = target.rpartition(":")
    if not separator or not host or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("--bvh-target must be HOST:PORT (IPv6: [ADDRESS]:PORT)")
    return host.strip("[]"), int(port)


class BvhSender:
    """Latest-frame sender: reconnects never block capture or inference."""
    def __init__(self, target, transport, stop):
        self.target, self.transport, self.stop = target, transport, stop
        self.address = parse_target(target) if target else None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending = None
        self.sent = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, name="bvh-network", daemon=True)

    def submit(self, metadata, frame):
        if self.address:
            with self.lock:
                self.pending = (metadata, frame, time.monotonic())
            self.wake.set()

    def stats(self):
        with self.lock:
            return dict(enabled=bool(self.address), target=self.target, transport=self.transport,
                        sent=self.sent, error=self.error, protocol="visual-mocap-bvh/1")

    def run(self):
        if not self.address:
            return
        connection, last_header, known_id = None, 0.0, None
        while not self.stop.is_set():
            self.wake.wait(.2)
            self.wake.clear()
            with self.lock:
                pending, self.pending = self.pending, None
            if not pending:
                continue
            metadata, frame, queued = pending
            try:
                if connection is None:
                    kind = socket.SOCK_DGRAM if self.transport == "udp" else socket.SOCK_STREAM
                    family, kind, protocol, _, remote = socket.getaddrinfo(*self.address, type=kind)[0]
                    connection = socket.socket(family, kind, protocol)
                    connection.settimeout(1)
                    connection.connect(remote)
                    last_header, known_id = 0.0, None
                if time.monotonic() - queued > .5:
                    continue
                messages = []
                identity = (metadata["stream_id"], metadata["skeleton_id"]) if metadata else None
                if metadata and (identity != known_id or self.transport == "udp" and time.monotonic()-last_header > 2):
                    messages.append(metadata)
                messages.append(frame)
                for message in messages:
                    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
                    if self.transport == "udp":
                        if len(payload) > 60000:
                            raise ValueError("BVH datagram exceeds maximum size")
                        connection.send(payload)
                    else:
                        connection.sendall(payload + b"\n")
                    if message is metadata:
                        last_header, known_id = time.monotonic(), identity
                with self.lock:
                    self.sent += 1
                    self.error = None
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.error = str(exc)
                if connection:
                    connection.close()
                    connection = None
                self.stop.wait(.5)
        if connection:
            connection.close()


class CalibrationSession:
    def __init__(self, args):
        cols, rows = map(int, args.board.split("x"))
        if (cols * rows) // 2 > 50:
            raise ValueError("Board is too large for DICT_4X4_50")
        self.board = cv2.aruco.CharucoBoard((cols, rows), args.square_m, args.marker_m,
                                          cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.samples = []
        self.lock = threading.Lock()
        self.last_sequences = None

    def capture(self, frames, all_ids):
        with self.lock:
            if set(frames) != set(all_ids):
                raise ValueError("All cameras need fresh frames within the time tolerance")
            sequences = tuple(frames[name].sequence for name in all_ids)
            if sequences == self.last_sequences:
                raise ValueError("These frames were already captured")
            if len(self.samples) >= 60:
                raise ValueError("Maximum 60 calibration samples; solve or reset")
            sample = {}
            for name, frame in frames.items():
                corners, ids, _, _ = self.detector.detectBoard(frame.image)
                if ids is None or len(ids) < 8:
                    raise ValueError(f"{name}: at least 8 ChArUco corners must be visible")
                sample[name] = dict(ids=ids.reshape(-1), corners=corners.reshape(-1, 2),
                                    size=(frame.image.shape[1], frame.image.shape[0]))
            self.samples.append(sample)
            self.last_sequences = sequences
            return len(self.samples)

    def solve(self, all_ids):
        with self.lock:
            if len(self.samples) < 12:
                raise ValueError("Capture at least 12 distinct board poses visible in all cameras")
            objects = self.board.getChessboardCorners()
            cameras, rms = {}, {}
            for name in all_ids:
                views = [sample[name] for sample in self.samples]
                size = views[0]["size"]
                if any(v["size"] != size for v in views):
                    raise ValueError(f"{name}: resolution changed during calibration")
                object_points = [objects[v["ids"]].astype(np.float32) for v in views]
                image_points = [v["corners"].astype(np.float32) for v in views]
                error, k, d, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
                if not math.isfinite(error) or error > 3:
                    raise ValueError(f"{name}: calibration RMS {error:.2f}px > 3px. Capture more varied, sharp board poses.")
                cameras[name] = dict(K=k.tolist(), D=d.reshape(-1).tolist(), R=np.eye(3).tolist(), t=[0., 0., 0.], image_size=list(size))
                rms[name] = float(error)
            reference = all_ids[0]
            for name in all_ids[1:]:
                objects_shared, left, right = [], [], []
                for sample in self.samples:
                    a, b = sample[reference], sample[name]
                    common, ia, ib = np.intersect1d(a["ids"], b["ids"], return_indices=True)
                    if len(common) >= 8:
                        objects_shared.append(objects[common].astype(np.float32))
                        left.append(a["corners"][ia].astype(np.float32))
                        right.append(b["corners"][ib].astype(np.float32))
                if len(objects_shared) < 12:
                    raise ValueError(f"{reference}/{name}: need 12 poses with 8 common corners")
                a, b = cameras[reference], cameras[name]
                error, _, _, _, _, r, t, _, _ = cv2.stereoCalibrate(
                    objects_shared, left, right, np.array(a["K"]), np.array(a["D"]),
                    np.array(b["K"]), np.array(b["D"]), tuple(a["image_size"]), flags=cv2.CALIB_FIX_INTRINSIC,
                    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-7))
                if not math.isfinite(error) or error > 3:
                    raise ValueError(f"{reference}/{name}: stereo RMS {error:.2f}px > 3px; check timing and board poses")
                b.update(R=r.tolist(), t=t.reshape(-1).tolist())
                rms[f"{reference}/{name}"] = float(error)
            return dict(schema_version=1, units="m", extrinsics_convention="world_to_camera",
                        world_frame=f"{reference}: X right, Y down, Z forward", cameras=cameras,
                        rms_px=rms, board=dict(squares=list(self.board.getChessboardSize()),
                        square_m=float(self.board.getSquareLength()), marker_m=float(self.board.getMarkerLength())))


class Engine:
    def __init__(self, args):
        self.args = args
        self.sources = camera_sources(args)
        self.ids = [s.id for s in self.sources]
        self.triangulate = load_triangulator(args.repos)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.calibrator = CalibrationSession(args)
        self.calibration_action_lock = threading.Lock()
        self.bvh_lock = threading.Lock()
        self.bvh = BvhAdapter(args)
        self.armature = ArmatureView(args.max_age_ms/1000)
        self.sender = BvhSender(args.bvh_target, args.bvh_transport, self.stop)
        self.calibration_data = demo_calibration(self.ids) if args.demo else None
        if not args.demo and args.calibration.exists():
            self.calibration_data = json.loads(args.calibration.read_text())
        self.calibration = checked_calibration(self.calibration_data, self.ids) if self.calibration_data else None
        self.cameras = {s.id: Camera(s, self.stop, args.preview_width, self.calibration[s.id] if args.demo else None) for s in self.sources}
        self.result = None
        self.processed = {}
        self.model_status = "synthetic" if args.demo else "loading"
        self.model_error = None
        self.providers = {}
        self.thread = threading.Thread(target=self.run, name="pose-inference", daemon=True)

    def start(self):
        for camera in self.cameras.values():
            camera.thread.start()
        self.thread.start()
        self.sender.thread.start()

    def close(self):
        self.stop.set()
        deadline = time.monotonic() + 6
        self.sender.wake.set()
        for thread in [*(c.thread for c in self.cameras.values()), self.thread, self.sender.thread]:
            if thread.ident:
                thread.join(max(0, deadline-time.monotonic()))
        with self.bvh_lock:
            self.bvh.close()

    def group(self):
        return select_frames({name: c.history() for name, c in self.cameras.items()}, time.monotonic(),
                             self.args.max_age_ms/1000, self.args.max_skew_ms/1000)

    def snapshot(self):
        now = time.monotonic()
        with self.bvh_lock:
            armature = self.armature.snapshot(now)
        with self.lock:
            result = self.result
            age = (now - result["completed_monotonic_s"])*1000 if result else None
            # A disconnected camera or crashed inference must not leave a
            # frozen pose looking like current tracking data.
            pose = result if age is not None and age < self.args.max_age_ms else None
            return dict(schema_version=1, demo=self.args.demo, model=f"RTMPose {self.args.preset}",
                        device="synthetic" if self.args.demo else self.args.device,
                        model_status=self.model_status, error=self.model_error, providers=self.providers,
                        calibrated=self.calibration is not None, calibration_samples=len(self.calibrator.samples),
                        calibration_rms_px=(self.calibration_data or {}).get("rms_px"),
                        world_frame=(self.calibration_data or {}).get("world_frame"),
                        timing_basis="server_monotonic_after_decode_plus_manual_offset",
                        max_skew_ms=self.args.max_skew_ms, pose_age_ms=age,
                        board=dict(squares=self.args.board, square_m=self.args.square_m, marker_m=self.args.marker_m),
                        bvh=dict(**self.sender.stats(), pose_status=self.bvh.status, file_error=self.bvh.file_error),
                        cameras=[c.stats(now) for c in self.cameras.values()], pose=pose, armature=armature,
                        joint_names=JOINTS, edges=EDGES)

    def preview(self, name, annotated):
        camera = self.cameras[name]
        with self.lock:
            frame = self.processed.get(name) if annotated else None
        if frame and time.monotonic() - frame[1] < 1:
            return frame
        with camera.lock:
            return camera.preview

    def run(self):
        try:
            model = None if self.args.demo else PoseModel(self.args)
            with self.lock:
                self.model_status = "synthetic" if self.args.demo else "ready"
                self.providers = model.providers if model else {}
            previous, counter = None, 0
            while not self.stop.is_set():
                started = time.monotonic()
                group = self.group()
                # Even with bad synchronization, show 2D detections for each
                # live camera; 3D uses only the coherent group below.
                frames = dict(group)
                for name, camera in self.cameras.items():
                    history = camera.history()
                    if name not in frames and history and started-history[-1].received < self.args.max_age_ms/1000:
                        frames[name] = history[-1]
                key = tuple((name, f.sequence) for name, f in sorted(frames.items()))
                if not frames or key == previous:
                    self.stop.wait(.01)
                    continue
                previous = key
                observations, views, previews = {}, {}, {}
                for name, frame in frames.items():
                    if model:
                        points, scores, people = model.infer(frame)
                    else:
                        points, scores, people = frame.synthetic_points, np.ones(17), 1
                    observations[name] = points, scores
                    views[name] = dict(frame_id=frame.sequence, receive_monotonic_s=frame.received,
                                       aligned_monotonic_s=frame.aligned, people=people,
                                       points=points.tolist(), confidence=scores.tolist(),
                                       image_size=[frame.image.shape[1], frame.image.shape[0]])
                    previews[name] = (frame.sequence, frame.received,
                        jpeg(overlay(frame.image, points, scores, self.args.confidence), self.args.preview_width))
                with self.lock:
                    calibration = self.calibration
                reconstruction = dict(points=[None]*17, confidence=[0.0]*17, reprojection_px=[None]*17)
                reason = "calibration_required"
                eligible = {}
                if calibration:
                    for name in group:
                        if tuple(views[name]["image_size"]) == calibration[name]["image_size"]:
                            eligible[name] = observations[name]
                    reason = "waiting_for_synchronized_views"
                    if len(eligible) >= 2:
                        reconstruction = reconstruct(eligible, calibration, self.triangulate,
                                                     self.args.confidence, self.args.max_reprojection_px)
                        reason = "tracking" if any(p is not None for p in reconstruction["points"]) else "no_valid_single_person_pose"
                    if len(eligible) < len(group):
                        reason = "calibration_resolution_mismatch"
                completed = time.monotonic()
                counter += 1
                with self.bvh_lock:
                    if self.stop.is_set():
                        break
                    with self.lock:
                        if calibration is not self.calibration:
                            continue
                    timestamp = max(f.received for f in group.values()) if group else completed
                    bvh_frame = self.bvh.update(reconstruction, timestamp, counter)
                    self.armature.update(self.bvh, bvh_frame, time.monotonic())
                    if bvh_frame:
                        bvh_frame["source_frame_ids"] = {name: frames[name].sequence for name in eligible}
                        bvh_frame["time_basis"] = "server_receive_monotonic"
                    message = bvh_frame or dict(protocol="visual-mocap-bvh/1", type="tracking_lost",
                        stream_id=self.bvh.stream_id, frame_id=counter, timestamp_monotonic_s=timestamp,
                        reason=self.bvh.status)
                    self.sender.submit(self.bvh.metadata, message)
                result = dict(frame_id=counter, completed_monotonic_s=completed,
                              receive_to_result_ms=(completed-min(f.received for f in frames.values()))*1000,
                              inference_cycle_ms=(completed-started)*1000,
                              reception_skew_ms=(max(f.aligned for f in group.values())-min(f.aligned for f in group.values()))*1000 if group else None,
                              triangulation_cameras=list(eligible), status=reason, units="m",
                              views=views, skeleton=reconstruction, bvh=bvh_frame)
                with self.lock:
                    self.result = result
                    self.processed.update(previews)
                self.stop.wait(max(0, 1/self.args.fps-(time.monotonic()-started)))
        except Exception as exc:
            LOG.exception("Inference stopped")
            with self.lock:
                self.model_status = "error"
                self.model_error = str(exc)
                self.result = None
                self.processed.clear()


def create_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        try:
            yield
        finally:
            await asyncio.to_thread(engine.close)

    app = FastAPI(title="Visual MoCap", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    @app.get("/healthz")
    def health():
        return {"model_status": engine.snapshot()["model_status"], "running": not engine.stop.is_set()}

    @app.get("/api/bvh/skeleton")
    def bvh_skeleton():
        with engine.bvh_lock:
            if not engine.bvh.metadata:
                raise HTTPException(409, engine.bvh.status)
            return engine.bvh.metadata

    @app.get("/video/{name}.mjpg")
    async def video(name: str, overlay: bool = True):
        if name not in engine.cameras:
            raise HTTPException(404, "Unknown camera")

        async def chunks():
            previous = None
            while not engine.stop.is_set():
                frame = engine.preview(name, overlay)
                if frame and (frame[0], frame[1]) != previous:
                    previous = frame[0], frame[1]
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame[2])).encode() + b"\r\n\r\n" + frame[2] + b"\r\n"
                await asyncio.sleep(1/30)
        return StreamingResponse(chunks(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        try:
            while not engine.stop.is_set():
                # No per-client queue. Slow viewers receive the latest state.
                await asyncio.wait_for(ws.send_json(engine.snapshot()), timeout=3)
                await asyncio.sleep(.05)
        except (WebSocketDisconnect, TimeoutError, OSError, RuntimeError):
            pass

    @app.get("/api/calibration/board.png")
    def board():
        cols, rows = engine.calibrator.board.getChessboardSize()
        image = engine.calibrator.board.generateImage((cols*160+80, rows*160+80), marginSize=40)
        _, data = cv2.imencode(".png", image)
        return Response(data.tobytes(), media_type="image/png")

    @app.post("/api/calibration/{action}")
    def calibrate(action: str):
        if engine.args.demo:
            raise HTTPException(409, "Calibration is disabled for synthetic demo")
        if not engine.calibration_action_lock.acquire(blocking=False):
            raise HTTPException(409, "Another calibration operation is running")
        try:
            if action == "capture":
                return {"samples": engine.calibrator.capture(engine.group(), engine.ids)}
            if action == "reset":
                with engine.calibrator.lock:
                    engine.calibrator.samples.clear()
                    engine.calibrator.last_sequences = None
                return {"samples": 0}
            if action == "solve":
                data = engine.calibrator.solve(engine.ids)
                calibration = checked_calibration(data, engine.ids)
                path = engine.args.calibration
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
                with engine.bvh_lock:
                    engine.bvh.close()
                    engine.bvh = BvhAdapter(engine.args)
                    engine.armature = ArmatureView(engine.args.max_age_ms/1000)
                    with engine.lock:
                        engine.calibration_data, engine.calibration = data, calibration
                        engine.result = None
                return {"saved": True, "rms_px": data["rms_px"]}
            raise HTTPException(404, "Unknown calibration action")
        except (ValueError, cv2.error) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            engine.calibration_action_lock.release()
    return app


HTML = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual MoCap · Live</title>
<style>
:root{color-scheme:light;font:15px system-ui;background:#f1f5f9;color:#1e293b}*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;border-bottom:1px solid #dbe3ed;background:#fff;display:flex;gap:20px;align-items:center;justify-content:space-between}h1{font-size:22px;margin:0}small,.muted{color:#64748b}main{padding:22px;max-width:1800px;margin:auto}.badge{border-radius:20px;padding:6px 12px;background:#e8eef5;font-size:13px}.good{color:#166534}.warn{color:#92400e}.layout{display:grid;grid-template-columns:minmax(280px,1fr) minmax(360px,1.4fr);gap:18px}.card{background:#fff;border:1px solid #dbe3ed;border-radius:12px;overflow:hidden;margin-bottom:16px}.bar{padding:12px 16px;display:flex;gap:12px;justify-content:space-between;align-items:center}img{width:100%;display:block;min-height:100px;background:#e2e8f0}.camera-stat{font-size:12px;color:#475569}.viewer{height:620px;position:relative;background:#f8fafc}canvas{display:block;width:100%;height:100%;touch-action:none}.hint{position:absolute;bottom:16px;left:18px;right:18px;color:#64748b;font-size:12px;pointer-events:none}.empty{position:absolute;top:16px;left:18px;right:18px;color:#475569;pointer-events:none;font-size:13px}.empty.live{color:#166534}.empty.held{color:#92400e}.stats{display:flex;flex-wrap:wrap;gap:24px;padding:16px;border-top:1px solid #dbe3ed}.stat b{display:block;font-size:24px;font-weight:500}.stat span{font-size:12px;color:#64748b}button,a.button{background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;border-radius:7px;padding:9px 12px;cursor:pointer;text-decoration:none;font:inherit}button:hover{background:#dbeafe}button:disabled{opacity:.5;cursor:wait}details{padding:16px;border-top:1px solid #dbe3ed}summary{cursor:pointer}p{line-height:1.55}.controls{display:flex;gap:8px;flex-wrap:wrap}#message{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.error{color:#b91c1c}#notice{margin:0 0 16px;padding:12px 16px;background:#e8eef5;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere}label{font-size:13px}@media(max-width:900px){.layout{grid-template-columns:1fr}.viewer{height:430px}header{padding:18px}main{padding:12px}}
</style>
<header><div><h1>Visual MoCap <span class="muted">/ Live</span></h1><small>Камеры → RTMPose → 3D</small></div><span id="connection" class="badge">Подключение…</span></header>
<main><div id="notice">Ожидание состояния сервера…</div><div class="layout"><section><div class="bar"><b>Видеопотоки</b><label><input id="overlay" type="checkbox" checked> Ключевые точки</label></div><div id="cameras"></div></section>
<section><div class="card"><div class="bar"><b>BVH-арматура · WebGL</b><button id="resetView">Центрировать</button></div><div class="viewer"><canvas id="gl"></canvas><div id="empty" class="empty">T-поза · ожидание BVH</div><div class="hint">Мышь / касание — вращать · колесо / щипок — масштаб · координаты в метрах</div></div><div class="stats"><div class="stat"><b id="fps">—</b><span>результатов / с</span></div><div class="stat"><b id="latency">—</b><span>приём → результат, мс</span></div><div class="stat"><b id="skew">—</b><span>разница времени приёма, мс</span></div><div class="stat"><b id="joints">0 / 17</b><span>3D-точек</span></div></div>
<details><summary>Калибровка камер</summary><p>Распечатайте ChArUco-доску. <span id="boardSize"></span> Соберите минимум 12 разных положений и наклонов. Доска должна быть неподвижна во время снимка и видна всем камерам.</p><div class="controls"><a class="button" href="/api/calibration/board.png" target="_blank" rel="noopener">Скачать доску</a><button data-action="capture">Добавить снимок</button><button data-action="solve">Вычислить и сохранить</button><button data-action="reset">Сбросить снимки</button></div><p>Снимков: <b id="samples">0</b></p><p id="message" role="status"></p></details><details open><summary>BVH</summary><p id="bvhStatus">Ожидание скелета…</p><p class="muted">Первые 20 пригодных поз задают длины костей (настраивается при запуске). Вращения конечностей вокруг их осей не измерены; кисти и стопы наследуют ориентацию родителя.</p></details></div><p class="muted">Режим одного человека. 3D использует калибровку и минимум два согласованных вида. Время приёма RTSP не равно времени экспозиции камеры.</p><p id="world" class="muted"></p></section></div></main>
<script>
'use strict';
const $=id=>document.getElementById(id), cards=new Map();let current=null, ws=null, seenFrame=-1, arrivals=[];
let armature=__REST_ARMATURE__, connected=false, lastStateMs=0;
const statuses={calibration_required:'Для 3D нужна калибровка камер',waiting_for_synchronized_views:'Ожидание двух согласованных видеопотоков',no_valid_single_person_pose:'Нет уверенной позы одного человека в двух камерах',calibration_resolution_mismatch:'Разрешение видео отличается от калибровки',tracking:''};
function setFeed(card,id){card.querySelector('img').src='/video/'+encodeURIComponent(id)+'.mjpg?overlay='+$('overlay').checked}
$('overlay').onchange=()=>cards.forEach((card,id)=>setFeed(card,id));
function update(s){current=s;lastStateMs=performance.now();if(s.armature)armature=s.armature; const pose=s.pose; let warnings=[];
 if(s.demo)warnings.push('ДЕМО: синтетические камеры и точки. Нейросетевой инференс не выполняется.');
 if(s.model_status==='loading')warnings.push('Загрузка модели и весов. Видео доступно независимо от инференса.');
 if(s.error)warnings.push('Ошибка инференса: '+s.error);
 if(!s.demo)warnings.push(s.model+' · '+s.device+' · '+s.model_status);
 $('notice').textContent=warnings.join('\n');$('samples').textContent=s.calibration_samples;
 $('boardSize').textContent=s.board.squares+' клеток; сторона клетки '+(s.board.square_m*1000).toFixed(1)+' мм, маркера '+(s.board.marker_m*1000).toFixed(1)+' мм.';
 $('bvhStatus').textContent=s.bvh.pose_status+(s.bvh.enabled?' · '+s.bvh.transport.toUpperCase()+' → '+s.bvh.target+' · отправлено '+s.bvh.sent:' · сетевой адрес не задан')+(s.bvh.error?' · '+s.bvh.error:'')+(s.bvh.file_error?' · запись: '+s.bvh.file_error:'');
 $('world').textContent=s.world_frame?'Система координат: '+s.world_frame+'. В WebGL оси Y и Z развёрнуты для отображения.':'';
 for(const c of s.cameras){let card=cards.get(c.id);if(!card){card=document.createElement('div');card.className='card';const bar=document.createElement('div');bar.className='bar';const name=document.createElement('b');name.textContent=c.id;const stat=document.createElement('span');stat.className='camera-stat';bar.append(name,stat);const img=document.createElement('img');img.alt=c.id;card.append(bar,img);$('cameras').append(card);cards.set(c.id,card);setFeed(card,c.id)}
 const people=pose?.views[c.id]?.people;card.querySelector('.camera-stat').textContent=c.status+' · '+c.fps+' fps'+(c.age_ms!==null?' · '+Math.round(c.age_ms)+' мс':'')+(people>1?' · несколько людей':'');}
 const now=performance.now();if(pose&&pose.frame_id!==seenFrame){seenFrame=pose.frame_id;arrivals.push(now)}arrivals=arrivals.filter(t=>now-t<2000);$('fps').textContent=arrivals.length>1?((arrivals.length-1)*1000/(now-arrivals[0])).toFixed(1):'—';
 $('latency').textContent=pose?pose.receive_to_result_ms.toFixed(0):'—';$('skew').textContent=pose?.reception_skew_ms!=null?pose.reception_skew_ms.toFixed(1):'—';
 const count=pose?pose.skeleton.points.filter(Boolean).length:0;$('joints').textContent=count+' / 17';
 for(const b of document.querySelectorAll('[data-action]'))b.disabled=s.demo;
}
function connect(){ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');ws.onopen=()=>{connected=true;$('connection').textContent='● Подключено';$('connection').className='badge good'};ws.onmessage=e=>update(JSON.parse(e.data));ws.onclose=()=>{connected=false;$('connection').textContent='Переподключение…';$('connection').className='badge warn';current=null;setTimeout(connect,1500)};ws.onerror=()=>ws.close()}
for(const button of document.querySelectorAll('[data-action]'))button.onclick=async()=>{button.disabled=true;$('message').textContent='Обработка…';try{const r=await fetch('/api/calibration/'+button.dataset.action,{method:'POST'}),v=await r.json();if(!r.ok)throw Error(v.detail||r.statusText);$('message').textContent=v.saved?'Калибровка сохранена. RMS: '+JSON.stringify(v.rms_px):'Снимков: '+v.samples}catch(e){$('message').textContent=e.message}finally{button.disabled=false}};
// Native WebGL, no CDN, Node build, Three.js or browser plugins required.
const canvas=$('gl'), gl=canvas.getContext('webgl',{antialias:true});let yaw=.35,pitch=.12,distance=3.5,target=[0,.9,0],autoCenter=true;
if(!gl){$('empty').textContent='WebGL недоступен. Включите аппаратное ускорение браузера.'}else{
 const vs='attribute vec3 p; attribute vec3 color; uniform mat4 mvp; varying vec3 c; void main(){gl_Position=mvp*vec4(p,1.0);gl_PointSize=7.0;c=color;}',fs='precision mediump float; varying vec3 c; void main(){gl_FragColor=vec4(c,1.0);}';
 function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,offset] of [['p',0],['color',12]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,24,offset)}const matrix=gl.getUniformLocation(program,'mvp');gl.enable(gl.DEPTH_TEST);
 const sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const n=Math.hypot(...a);return a.map(x=>x/n)};
 function multiply(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
 function mvp(){const eye=[target[0]+distance*Math.cos(pitch)*Math.sin(yaw),target[1]+distance*Math.sin(pitch),target[2]+distance*Math.cos(pitch)*Math.cos(yaw)],z=norm(sub(eye,target)),x=norm(cross([0,1,0],z)),y=cross(z,x);const view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];const f=1/Math.tan(Math.PI/8),aspect=canvas.width/canvas.height,n=.02,far=150;return multiply([f/aspect,0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0],view)}
 const pointers=new Map();canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY])};canvas.onpointerup=canvas.onpointercancel=e=>pointers.delete(e.pointerId);canvas.onpointermove=e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId);if(pointers.size===1){yaw-=(e.clientX-old[0])*.006;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-old[1])*.006))}else{const other=[...pointers.entries()].find(([id])=>id!==e.pointerId)[1];const before=Math.hypot(old[0]-other[0],old[1]-other[1]),after=Math.hypot(e.clientX-other[0],e.clientY-other[1]);if(after>0)distance=Math.max(.3,Math.min(80,distance*before/after))}pointers.set(e.pointerId,[e.clientX,e.clientY])};canvas.onwheel=e=>{e.preventDefault();distance=Math.max(.3,Math.min(80,distance*Math.exp(e.deltaY*.001))) };
 $('resetView').onclick=()=>{autoCenter=true;yaw=.35;pitch=.12;distance=3.5};
 // Octahedra match host/viz/pyqt_mocap/human_canvas_gl.py: 20% head,
 // radius 12% (at most 4.5 cm), eight shaded facets and dark edges.
 const add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,k)=>a.map(v=>v*k);
 const shades=[1,.90,.78,.70,.88,.80,.96,.85];
 function boneMesh(bone,color,triangles,edges){
  const delta=sub(bone.end,bone.start),length=Math.hypot(...delta);if(length<1e-8)return;
  const direction=scale(delta,1/length),axes=bone.axes;
  const reference=axes.reduce((a,b)=>Math.abs(dot(direction,a))<=Math.abs(dot(direction,b))?a:b);
  const u=norm(cross(direction,reference)),v=cross(direction,u),radius=Math.min(length*.12,.045),center=add(bone.start,scale(delta,.2));
  const vertices=[bone.start,bone.end,add(center,scale(u,radius)),add(center,scale(v,radius)),add(center,scale(u,-radius)),add(center,scale(v,-radius))];
  for(let i=0;i<4;i++){const a=2+i,b=2+(i+1)%4;for(const [k,face] of [[0,[0,b,a]],[1,[1,a,b]]])for(const index of face)triangles.push(...vertices[index],...scale(color,shades[2*i+k]));
   for(const [a,b] of [[0,2+i],[1,2+i],[2+i,2+(i+1)%4]])edges.push(...vertices[a],.18,.24,.30,...vertices[b],.18,.24,.30);
  }
 }
 let centeredRig=null,centeredLive=false,floor=0;
 function draw(){
  const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.viewport(0,0,w,h);gl.clearColor(248/255,250/255,252/255,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const age=(armature.age_ms||0)+performance.now()-lastStateMs;
  const live=connected&&armature.status==='live'&&age<=armature.max_age_ms;
  const held=armature.frame_id!==null&&!live;
  $('empty').className='empty'+(live?' live':held?' held':'');
  $('empty').textContent=live?'● BVH · движение'+(armature.bones.some(b=>!b.tracked&&!b.name.endsWith(':End'))?' · часть костей удерживается':''):held?'Поза удерживается · '+(!connected?'связь потеряна':'нет стабильного свежего BVH'):armature.status==='waiting'?'T-поза · ожидание 3 последовательных BVH-кадров':'T-поза · '+(current?.pose?(statuses[current.pose.status]||'ожидание BVH'):'ожидание BVH');
  if(centeredRig!==armature.rig_id){centeredRig=armature.rig_id;centeredLive=false;autoCenter=true}
  if(armature.frame_id!==null&&!centeredLive){centeredLive=true;autoCenter=true}
  const points=armature.points;
  if(autoCenter){target=[0,1,2].map(i=>points.reduce((sum,p)=>sum+p[i],0)/points.length);floor=Math.min(...points.map(p=>p[1]));autoCenter=false}
  const grid=[],triangles=[],edges=[];
  const line=(a,b,c)=>grid.push(...a,...c,...b,...c);
  for(let i=-5;i<=5;i++){line([target[0]+i*.5,floor,target[2]-2.5],[target[0]+i*.5,floor,target[2]+2.5],[.82,.86,.90]);line([target[0]-2.5,floor,target[2]+i*.5],[target[0]+2.5,floor,target[2]+i*.5],[.82,.86,.90])}
  for(const bone of armature.bones){const color=live&&bone.tracked?(bone.name.includes('Leg')||bone.name.includes('Foot')?[.30,.58,.77]:[.88,.55,.41]):held?[.66,.69,.73]:[.54,.62,.70];boneMesh(bone,color,triangles,edges)}
  gl.uniformMatrix4fv(matrix,false,mvp());
  function upload(data,mode){gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);gl.drawArrays(mode,0,data.length/6)}
  upload(grid,gl.LINES);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);upload(triangles,gl.TRIANGLES);gl.disable(gl.POLYGON_OFFSET_FILL);upload(edges,gl.LINES);
  requestAnimationFrame(draw)
 }draw();
}connect();
</script></html>'''.replace('__REST_ARMATURE__', json.dumps(ArmatureView(.5).snapshot(0)))


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{args.rtsp_transport}")
    engine = Engine(args)
    import uvicorn
    LOG.info("Open http://%s:%s (use server IP when binding 0.0.0.0)", args.host, args.port)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, access_log=False, workers=1)


if __name__ == "__main__":
    try:
        main(CLI_ARGS)
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(str(exc))
