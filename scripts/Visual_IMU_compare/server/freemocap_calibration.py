#!/usr/bin/env python3
"""Headless adapter to FreeMoCap's unchanged Anipose calibration solver.

Run in .venv-calibration. Video decoding/detection and timestamp matching live
in visual_mocap_server.py; this process only adapts observations and results.
No calibration algorithm is implemented here and no fallback solver is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def load_native(repos, log_folder):
    repos = Path(repos).resolve()
    # Some upstream repos do not ignore __pycache__; keep pinned checkouts clean.
    sys.dont_write_bytecode = True
    names = ("freemocap", "skellycam", "skellytracker", "skellylogs")
    for name in names:
        if not (repos / name / name / "__init__.py").is_file():
            raise ValueError(f"Missing {name} checkout in {repos}; rerun server/setup.sh")
    sys.path[:0] = [str(repos / name) for name in names]
    os.environ["FREEMOCAP_BASE_FOLDER"] = str(Path(log_folder).resolve())
    # Avoid competing with RTMPose for all CPU cores during bundle adjustment.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import freemocap
    from freemocap.core.tasks.calibration.anipose_calibration.helpers import bundle_adjust
    from freemocap.core.tasks.calibration.shared.camera_model import CameraModel
    from freemocap.core.tasks.calibration.shared.camera_intrinsics import CameraIntrinsics
    from freemocap.core.tasks.calibration.shared.camera_extrinsics import CameraExtrinsics
    from skellytracker.core.detectors.keypoint_detectors.charuco import CharucoBoardDefinition
    commit = subprocess.check_output(
        ["git", "-C", str(repos / "freemocap"), "rev-parse", "HEAD"], text=True).strip()
    return (bundle_adjust, CameraModel, CameraIntrinsics, CameraExtrinsics, CharucoBoardDefinition,
            dict(name="freemocap_anipose", version=freemocap.__version__, commit=commit,
                 entrypoint=f"{bundle_adjust.__name__}.calibrate_cameras_from_rows"))


def convert_cameras(cameras):
    """Rebase world->camera extrinsics to camera 0, and millimetres to metres."""
    reference = cameras[0].extrinsics
    r0, t0 = reference.rotation_matrix, reference.translation
    result = {}
    for camera in cameras:
        r = camera.extrinsics.rotation_matrix @ r0.T
        t = (camera.extrinsics.translation - r @ t0) / 1000.0
        result[camera.id] = dict(K=camera.intrinsics.to_camera_matrix().tolist(),
            D=camera.intrinsics.to_dist_coeffs_5().tolist(), R=r.tolist(), t=t.tolist(),
            image_size=list(camera.image_size))
    return result


def solve(payload, native):
    import cv2
    import numpy as np
    solver, CameraModel, Intrinsics, Extrinsics, Board, metadata = native
    ids, samples = payload["camera_ids"], payload["samples"]
    if len(ids) < 2 or len(ids) != len(set(ids)) or len(samples) < 12:
        raise ValueError("FreeMoCap calibration needs >=2 unique cameras and >=12 board poses")
    definition = payload["board"]
    board = Board(squares_x=definition["squares"][0], squares_y=definition["squares"][1],
                  square_length_mm=definition["square_m"] * 1000.0,
                  marker_length_ratio=definition["marker_m"] / definition["square_m"],
                  aruco_dictionary_enum=cv2.aruco.DICT_4X4_50)
    cameras, all_rows = [], []
    for index, name in enumerate(ids):
        size = tuple(payload["image_sizes"][name])
        cameras.append(CameraModel(id=name, index=index, image_size=size,
            intrinsics=Intrinsics.from_image_size(width=size[0], height=size[1]),
            extrinsics=Extrinsics.identity()))
        rows = []
        for number, sample in enumerate(samples):
            view = sample[name]
            corner_ids = np.asarray(view["ids"], dtype=np.int32).reshape(-1)
            corners = np.asarray(view["corners"], dtype=np.float64).reshape(-1, 1, 2)
            if (len(corner_ids) != len(corners) or len(corner_ids) < 8
                    or len(set(corner_ids.tolist())) != len(corner_ids)
                    or np.any(corner_ids < 0) or np.any(corner_ids >= board.n_corners)
                    or not np.isfinite(corners).all()):
                raise ValueError(f"Invalid ChArUco observations for {name}, group {number}")
            # The same matched group number in every camera; not the AVI index.
            rows.append(dict(framenum=(0, number), ids=corner_ids, corners=corners))
        all_rows.append(rows)
    np.random.seed(0)  # upstream resampling is stochastic; reproducible per recording
    error, merged, _ = solver.calibrate_cameras_from_rows(cameras=cameras, all_rows=all_rows, board=board)
    if not math.isfinite(error) or error > 3:
        raise ValueError(f"FreeMoCap median reprojection error {error:.2f}px > 3px; record better board views")
    # Report true per-camera RMS separately from the solver's median statistic.
    points_2d, _ = solver.extract_points(merged, board, camera_ids=ids, min_cameras=2)
    points_3d = solver.triangulate(cameras, points_2d)
    residuals = solver.reprojection_error(cameras, points_3d, points_2d, mean=False)
    rms = {name: float(np.sqrt(np.nanmean(np.sum(residuals[i] ** 2, axis=-1))))
           for i, name in enumerate(ids)}
    if not all(math.isfinite(value) for value in rms.values()):
        raise ValueError("FreeMoCap returned nonfinite reprojection errors")
    return dict(schema_version=1, units="m", extrinsics_convention="world_to_camera",
                world_frame=f"{ids[0]}: X right, Y down, Z forward",
                cameras=convert_cameras(cameras), board=definition, rms_px=rms,
                solver=dict(metadata, median_reprojection_px=float(error)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Check native solver imports only")
    args = parser.parse_args(argv)
    if not args.check and (args.input is None or args.output is None):
        parser.error("--input and --output are required")
    # Normal runs keep upstream logs inside the temporary calibration directory.
    log_folder = args.output.parent / "freemocap-logs" if args.output else Path(__file__).parent / "runtime/calibration-worker"
    native = load_native(args.repos, log_folder)
    if args.check:
        print(json.dumps(native[-1]))
    else:
        result = solve(json.loads(args.input.read_text()), native)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
