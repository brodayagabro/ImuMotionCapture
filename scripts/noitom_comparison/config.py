"""Experiment options; the own-system settings remain in the existing viewer."""
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import sys

BVH_ROTATIONS = {"XYZ": 0, "XZY": 1, "YXZ": 2, "YZX": 3, "ZXY": 4, "ZYX": 5}


@dataclass(frozen=True)
class ComparisonConfig:
    sdk_library: str | None = None
    transport: str = "udp"
    port: int = 7012
    server: str = "127.0.0.1"
    bvh_format: str = "binary"
    bvh_rotation: str = "XYZ"
    avatar_index: int | None = None
    stale_timeout_ms: int = 500
    recordings_dir: str = "recordings"
    fake: bool = False
    event_port: int = 0

    def __post_init__(self):
        if self.transport not in {"udp", "tcp"}:
            raise ValueError("transport must be udp or tcp")
        if self.bvh_format not in {"binary", "string", "legacy"}:
            raise ValueError("bvh_format must be binary, string or legacy")
        if self.bvh_rotation not in BVH_ROTATIONS:
            raise ValueError("Unsupported BVH rotation order")
        if not 1 <= self.port <= 65535 or self.stale_timeout_ms <= 0:
            raise ValueError("Invalid port or stale timeout")
        if not 0 <= self.event_port <= 65535:
            raise ValueError("Invalid external event port")

    def library_path(self) -> Path:
        if self.sdk_library:
            return Path(self.sdk_library).expanduser().resolve()
        root = Path(__file__).resolve().parents[1] / "MocapApi"
        demo = root / "demo/demo-py/MocapApi/mocap_api/lib"
        if sys.platform == "win32":
            candidates = (demo / "amd64/MocapApi.dll", root / "bin/win32/x64/release/MocapApi.dll")
        else:
            arm = platform.machine().lower() in {"aarch64", "arm64"}
            candidates = (demo / ("arm64" if arm else "x86_64") / "libMocapApi.so",
                          root / "bin" / "linux" / ("aarch64" if arm else "x64") / "libMocapApi.so")
        return next((path for path in candidates if path.is_file()), candidates[0])

    def metadata(self):
        return {**asdict(self), "resolved_library": str(self.library_path())}
