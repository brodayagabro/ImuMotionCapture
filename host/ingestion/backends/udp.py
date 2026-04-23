import socket
from host.shared.data_parser import extract_floats
from host.ingestion.backends.base import BaseDataReader

class UDPReader(BaseDataReader):
    def __init__(self, port: int, data_callback, error_callback):
        super().__init__(data_callback, error_callback)
        self.port = port
        self.udp_socket = None
        self._buffer = b''

    def connect(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', self.port))
        self.udp_socket.settimeout(0.1)
        self.running = True

    def disconnect(self):
        self.running = False
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except Exception:
                pass

    def read_loop(self):
        while self.running:
            try:
                data, _ = self.udp_socket.recvfrom(1024)
                self._buffer += data
                while b'\r\n' in self._buffer or b'\n' in self._buffer:
                    sep = b'\r\n' if b'\r\n' in self._buffer else b'\n'
                    packet, self._buffer = self._buffer.split(sep, 1)
                    line = packet.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    if line.startswith('WT'):
                        line = line[12:]
                    values = extract_floats(line)
                    if values is not None:
                        self.data_callback(values[:3], values[3:6], values[6:9])
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.error_callback(f"UDP error: {e}")
                break