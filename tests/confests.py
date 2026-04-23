import sys
import pytest
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

# 🔧 Авто-добавление корня проекта (как в main.py)
CURRENT_FILE = Path(__file__).resolve()
for parent in [CURRENT_FILE] + list(CURRENT_FILE.parents):
    if (parent / 'host').is_dir() and (parent / 'firmware').is_dir():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="function")
def mock_serial():
    """Mock для serial.Serial."""
    with patch('host.ingestion.backends.serial.serial.Serial') as mock:
        instance = mock.return_value
        instance.in_waiting = 0
        instance.readline.return_value = b''
        instance.is_open = True
        yield instance


@pytest.fixture(scope="function")
def mock_udp_socket():
    """Mock для socket.socket."""
    with patch('host.ingestion.backends.udp.socket.socket') as mock:
        instance = mock.return_value
        instance.recvfrom.side_effect = socket.timeout()
        yield instance


@pytest.fixture(scope="function")
def headless_tk():
    """Запуск Tkinter в головном режиме (без отображения окон)."""
    # Для Linux: требуется 'export DISPLAY=:99' или xvfb-run
    # Для Windows/macOS работает из коробки с tkinter>=8.6
    root = tk.Tk()
    root.withdraw()  # Скрыть окно
    yield root
    root.destroy()


@pytest.fixture(scope="function")
def sample_imu_line():
    """Пример строки с данными датчика."""
    return "WT901C  0.12  -0.45  9.81  0.01  -0.02  0.00  23.4  -12.1  45.6\r\n"


@pytest.fixture(scope="function")
def sample_accel():
    return np.array([0.12, -0.45, 9.81])


@pytest.fixture(scope="function")
def sample_gyro():
    return np.array([0.01, -0.02, 0.00])


@pytest.fixture(scope="function")
def sample_mag():
    return np.array([23.4, -12.1, 45.6])