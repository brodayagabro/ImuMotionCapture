"""The supported viewer must work without conference or legacy render code."""
from pathlib import Path
import subprocess
import sys


def test_main_viewer_imports_without_conference_or_matplotlib():
    viz = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class BlockOptionalApps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('scripts', 'matplotlib', 'cube_viewer'):
            raise ImportError('Main viewer imported optional app: ' + fullname)
sys.meta_path.insert(0, BlockOptionalApps())
from pyqt_mocap.mpu_udp_viewer import MotionCaptureWindow
from pyqt_mocap.mpu_udp_viewer_gl import OpenGLMotionCaptureWindow
from pyqt_mocap.human_canvas_gl import OpenGLHumanCanvas
assert OpenGLMotionCaptureWindow is MotionCaptureWindow
assert MotionCaptureWindow.canvas_class is OpenGLHumanCanvas
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(viz)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
