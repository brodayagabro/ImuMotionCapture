import threading
from host.shared.config import DEFAULT_SERIAL_PORT, DEFAULT_UDP_PORT
from host.ingestion.backends.serial import SerialReader
from host.ingestion.backends.udp import UDPReader

class SensorCoordinator:
    def __init__(self, data_callback, error_callback):
        self.data_callback = data_callback
        self.error_callback = error_callback
        self.backend = None
        self.thread = None

    def connect_serial(self, port: str = None):
        port = port or DEFAULT_SERIAL_PORT
        self.backend = SerialReader(port, self.data_callback, self.error_callback)
        self.backend.connect()
        self._start_thread()

    def connect_udp(self, port: int = None):
        port = port or DEFAULT_UDP_PORT
        self.backend = UDPReader(port, self.data_callback, self.error_callback)
        self.backend.connect()
        self._start_thread()

    def _start_thread(self):
        self.thread = threading.Thread(target=self.backend.read_loop, daemon=True)
        self.thread.start()

    def disconnect(self):
        if self.backend:
            self.backend.disconnect()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.backend = None
        self.thread = None

    def send_command(self, command: str):
        if self.backend:
            self.backend.send_command(command)