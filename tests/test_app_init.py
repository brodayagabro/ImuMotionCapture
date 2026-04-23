import pytest
import tkinter as tk
from unittest.mock import patch, MagicMock
from host.viz.app import IMUApp


class TestIMUAppInit:
    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """Мокаем внешние зависимости перед каждым тестом."""
        with patch('host.viz.app.SensorCoordinator'), \
             patch('host.viz.app.AHRSProcessor'), \
             patch('host.viz.app.Visualizer'):
            yield

    def test_app_creates_widgets(self, headless_tk):
        app = IMUApp(headless_tk)
        assert app.root is not None
        assert app.control_frame is not None
        assert app.plot_frame is not None
        assert app.connect_btn is not None
        assert app.mode_cb is not None

    def test_initial_state(self, headless_tk):
        app = IMUApp(headless_tk)
        with app.data_lock:
            assert app.latest_data["connected"] is False
            assert app.latest_data["valid"] is False
            assert app.latest_data["packets_received"] == 0

    def test_mode_change_serial(self, headless_tk):
        app = IMUApp(headless_tk)
        app.mode_var.set("Serial")
        app._on_mode_change(None)
        assert app.port_entry.cget('state') == 'normal'
        assert app.udp_entry.cget('state') == 'disabled'

    def test_mode_change_udp(self, headless_tk):
        app = IMUApp(headless_tk)
        app.mode_var.set("UDP")
        app._on_mode_change(None)
        assert app.port_entry.cget('state') == 'disabled'
        assert app.udp_entry.cget('state') == 'normal'