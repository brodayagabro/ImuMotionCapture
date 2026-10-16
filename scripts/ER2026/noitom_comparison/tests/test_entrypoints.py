"""Relocation must not depend on the terminal's working directory."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ER2026.paths import EXPERIMENT_ROOT, PROJECT_ROOT, RECORDS_ROOT, SDK_ROOT
from scripts.ER2026.noitom_comparison.config import ComparisonConfig
from scripts.ER2026.experiments.semaphore.config import ROOT, load_config


@pytest.mark.parametrize('relative', [
    'main.py', 'noitom_comparison/main.py', 'experiments/semaphore/main.py',
])
def test_direct_launch_from_another_directory(tmp_path, relative):
    result = subprocess.run([sys.executable, '-I', str(EXPERIMENT_ROOT / relative), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


@pytest.mark.parametrize('module', [
    'scripts.ER2026', 'scripts.ER2026.experiments.semaphore.main',
    'scripts.ER2026.experiments.semaphore.dataset',
    'scripts.ER2026.diagnostics.orientation.calibration_analyzer',
    'scripts.ER2026.diagnostics.orientation.return_pose_check',
])
def test_module_launch(module):
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_default_locations_do_not_follow_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ComparisonConfig()
    assert Path(config.recordings_dir) == RECORDS_ROOT == PROJECT_ROOT / 'scripts' / 'ER2026' / 'records'
    assert ROOT / load_config().sessions_dir == RECORDS_ROOT
    assert config.library_path().is_relative_to(SDK_ROOT)
    assert SDK_ROOT == PROJECT_ROOT / 'scripts' / 'MocapApi'
