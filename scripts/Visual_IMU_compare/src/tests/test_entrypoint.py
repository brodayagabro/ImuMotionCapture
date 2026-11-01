from pathlib import Path
import os
import subprocess
import sys


def test_cli_from_other_directory_and_no_sdk_import(tmp_path):
    root=Path(__file__).resolve().parents[4]
    entry=root/'scripts/Visual_IMU_compare/src/main.py'
    result=subprocess.run([sys.executable,str(entry),'--help'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==0, result.stderr
    assert '--source-ip' in result.stdout
    code='''import sys
from scripts.Visual_IMU_compare.src import main
from scripts.Visual_IMU_compare.src.comparison_window import MocapComparisonWindow
assert 'scripts.ER2026.noitom_comparison.noitom_client' not in sys.modules
'''
    env=dict(os.environ,PYTHONPATH=str(root))
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0, result.stderr
