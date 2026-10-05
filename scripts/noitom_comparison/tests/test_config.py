from pathlib import Path
import pytest
from scripts.noitom_comparison.config import BVH_ROTATIONS, ComparisonConfig


def test_defaults_match_working_demo_and_prefer_its_dll(monkeypatch):
    monkeypatch.setattr("scripts.noitom_comparison.config.sys.platform", "win32")
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    config = ComparisonConfig()
    assert (config.transport, config.port, config.bvh_format, config.bvh_rotation) == ("udp", 7012, "binary", "XYZ")
    assert config.library_path().as_posix().endswith("demo/demo-py/MocapApi/mocap_api/lib/amd64/MocapApi.dll")
    assert BVH_ROTATIONS["XYZ"] == 0 and BVH_ROTATIONS["ZXY"] == 4


def test_explicit_library_and_legacy_fallback(monkeypatch, tmp_path):
    explicit = tmp_path / "custom.dll"
    assert ComparisonConfig(sdk_library=str(explicit)).library_path() == explicit.resolve()
    monkeypatch.setattr("scripts.noitom_comparison.config.sys.platform", "win32")
    monkeypatch.setattr(Path, "is_file", lambda path: "bin/win32" in path.as_posix())
    assert ComparisonConfig().library_path().as_posix().endswith("bin/win32/x64/release/MocapApi.dll")
    with pytest.raises(ValueError, match="rotation"):
        ComparisonConfig(bvh_rotation="invalid")
