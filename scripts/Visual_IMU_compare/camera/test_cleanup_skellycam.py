from pathlib import Path
import sys

import pytest

import cleanup_skellycam as cleanup


@pytest.fixture
def root(tmp_path, monkeypatch):
    for name in (".venv-skellycam", ".tools/recording-bootstrap", ".tools/python", "runtime/records"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "keep.txt").write_text("data")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    monkeypatch.setattr(cleanup, "check_inactive", lambda root: None)
    return tmp_path


def test_dry_run_and_cleanup_preserve_recordings_rtsp_and_shared_python(root):
    targets = cleanup.cleanup(root)
    assert len(targets) == 2 and all(path.exists() for path in targets)
    assert cleanup.cleanup(root, apply=True) == targets
    assert all(not path.exists() for path in targets)
    for name in (".tools/python/keep.txt", "runtime/records/keep.txt", ".venv/bin/python"):
        assert (root / name).is_file()


def test_symlink_outside_checkout_is_not_followed(root, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (root / "repositories").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cleanup.cleanup(root, apply=True)
    assert (root / ".venv-skellycam").exists()


def test_dependent_rtsp_python_blocks_all_deletions(root):
    old_python = root / ".venv-skellycam/python"
    old_python.write_bytes(b"not an interpreter")
    (root / ".venv/bin/python").unlink()
    (root / ".venv/bin/python").symlink_to(old_python)
    with pytest.raises(ValueError, match="depends"):
        cleanup.cleanup(root, apply=True)
    assert (root / ".tools/recording-bootstrap").exists()


def test_active_old_service_or_process_blocks_deletion(root, monkeypatch):
    def active(root):
        raise ValueError("Stop the old SkellyCam process first")
    monkeypatch.setattr(cleanup, "check_inactive", active)
    with pytest.raises(ValueError, match="Stop"):
        cleanup.cleanup(root, apply=True)
    assert (root / ".venv-skellycam").exists()


def test_modified_or_unknown_checkout_blocks_all_deletion(root, monkeypatch):
    repo = root / "repositories/skellycam"
    repo.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unrecognized"):
        cleanup.cleanup(root, apply=True)
    (repo / ".git").mkdir()
    monkeypatch.setattr(cleanup.subprocess, "check_output", lambda *args, **kwargs: " M local.py\n")
    with pytest.raises(ValueError, match="Local modifications"):
        cleanup.cleanup(root, apply=True)
    assert (root / ".venv-skellycam").exists()
