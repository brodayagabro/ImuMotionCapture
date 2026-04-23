import pytest
from unittest.mock import MagicMock, patch, call
from host.ingestion.backends.base import BaseDataReader
from host.ingestion.backends.serial import SerialReader
from host.ingestion.backends.udp import UDPReader


class TestBaseDataReader:
    def test_abstract_methods(self):
        def dummy_cb(data, err): pass
        with pytest.raises(TypeError):
            BaseDataReader(dummy_cb, dummy_cb)


class TestSerialReader:
    def test_connect_calls_serial_init(self, mock_serial):
        def dummy_cb(a, g, m): pass
        def dummy_err(msg): pass
        reader = SerialReader('COM3', dummy_cb, dummy_err)
        reader.connect()
        mock_serial.assert_called_once()

    def test_send_command_writes_to_serial(self, mock_serial):
        def dummy_cb(a, g, m): pass
        def dummy_err(msg): pass
        reader = SerialReader('COM3', dummy_cb, dummy_err)
        reader.ser = mock_serial
        reader.send_command("S100")
        mock_serial.write.assert_called_once_with(b'S100\n')

    def test_read_loop_skips_status_lines(self, mock_serial, sample_imu_line):
        received = []
        def on_data(accel, gyro, mag):
            received.append((accel, gyro, mag))
        def on_err(msg): pass
        
        reader = SerialReader('COM3', on_data, on_err)
        reader.ser = mock_serial
        mock_serial.in_waiting = 1
        mock_serial.readline.side_effect = [
            b'OK:START\r\n',
            sample_imu_line.encode(),
            b'ERR:Timeout\r\n',
            b''  # выход по пустой строке
        ]
        # Запускаем на 1 итерацию
        reader.running = True
        reader.read_loop()
        # Должен обработать только валидную строку
        assert len(received) == 1


class TestUDPReader:
    def test_connect_binds_socket(self, mock_udp_socket):
        def dummy_cb(a, g, m): pass
        def dummy_err(msg): pass
        reader = UDPReader(1399, dummy_cb, dummy_err)
        reader.connect()
        mock_udp_socket.bind.assert_called_once_with(('0.0.0.0', 1399))

    def test_read_loop_parses_wt_prefix(self, mock_udp_socket):
        received = []
        def on_data(accel, gyro, mag):
            received.append((accel, gyro, mag))
        def on_err(msg): pass
        
        reader = UDPReader(1399, on_data, on_err)
        reader.udp_socket = mock_udp_socket
        
        wt_line = b'WT901C  0.1 0.2 0.3 0.01 0.02 0.03 10 20 30\r\n'
        mock_udp_socket.recvfrom.return_value = (wt_line, ('127.0.0.1', 12345))
        
        reader.running = True
        # Один цикл обработки
        reader.read_loop()
        assert len(received) == 1