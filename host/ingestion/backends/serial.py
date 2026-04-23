import serial
import time
from host.shared.config import BAUD_RATE, TIMEOUT
from host.shared.data_parser import extract_floats
from host.ingestion.backends.base import BaseDataReader

class SerialReader(BaseDataReader):
    def __init__(self, port: str, data_callback, error_callback):
        super().__init__(data_callback, error_callback)
        self.port = port
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(self.port, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(1)
        self.running = True

    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'P\n')
                self.ser.close()
            except Exception:
                pass

    def send_command(self, command: str):
        if self.ser and self.ser.is_open:
            self.ser.write(f'{command}\n'.encode())
            self.ser.flush()

    def read_loop(self):
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line or line.startswith(('OK:', 'DONE:', 'ERR:', 'CAL:', 'STAT:', 'RDY', 'S')):
                        continue
                    values = extract_floats(line)
                    if values is not None:
                        self.data_callback(values[:3], values[3:6], values[6:9])
                time.sleep(0.005)
            except Exception as e:
                if self.running:
                    self.error_callback(f"Serial error: {e}")
                time.sleep(1)