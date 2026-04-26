import threading
import logging
from typing import Callable, Optional
from host.shared.config import DEFAULT_SERIAL_PORT, DEFAULT_UDP_PORT
from host.ingestion.backends.base import BaseDataReader
from host.ingestion.backends.serial import SerialReader
from host.ingestion.backends.udp import UDPReader

logger = logging.getLogger(__name__)


class SensorCoordinator:
    """
    Координатор чтения данных с датчиков.
    Управляет бэкендами (Serial/UDP), потоками и callback'ами.
    """
    def __init__(self, data_callback: Callable, error_callback: Callable, status_callback: Optional[Callable] = None):
        """
        Args:
            data_callback: Функция вызываемая при получении данных (accel, gyro, mag)
            error_callback: Функция для обработки ошибок (msg: str)
            status_callback: Опциональная функция для обновления статуса (msg: str)
        """
        self.data_callback = data_callback
        self.error_callback = error_callback
        self.status_callback = status_callback if status_callback else lambda msg: None

        self._lock = threading.Lock()
        self.backend: Optional[BaseDataReader] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.backend_type: Optional[str] = None

    def connect_serial(self, port: str = None) -> bool:
        """Подключение через COM-порт."""
        return self._connect_backend("serial", port=port or DEFAULT_SERIAL_PORT)

    def connect_udp(self, port: int = None) -> bool:
        """Подключение через UDP."""
        return self._connect_backend("udp", port=port or DEFAULT_UDP_PORT)

    def _connect_backend(self, backend_type: str, **kwargs) -> bool:
        """Внутренний метод подключения бэкенда."""
        with self._lock:
            if self.running:
                self.status_callback("⚠️ Already connected")
                return False

            try:
                if backend_type == "serial":
                    self.backend = SerialReader(
                        kwargs.get("port"), 
                        self.data_callback, 
                        self.error_callback
                    )
                elif backend_type == "udp":
                    self.backend = UDPReader(
                        kwargs.get("port"), 
                        self.data_callback, 
                        self.error_callback
                    )
                else:
                    self.error_callback(f" Unknown backend: {backend_type}")
                    return False

                self.backend.connect()
                self.backend_type = backend_type
                self.running = True

                self.thread = threading.Thread(
                    target=self._run_loop,
                    daemon=True,
                    name=f"Reader-{backend_type}"
                )
                self.thread.start()
                
                self.status_callback(f" Connected via {backend_type.upper()}")
                return True

            except Exception as e:
                self.error_callback(f" Connection failed: {e}")
                self._cleanup()
                return False

    def disconnect(self) -> None:
        """Отключение и остановка потока."""
        with self._lock:
            if not self.running:
                return
            self.running = False

        if self.backend:
            try:
                self.backend.disconnect()
            except Exception as e:
                self.error_callback(f" Disconnect error: {e}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self._lock:
            self._cleanup()
            self.status_callback(" Disconnected")

    def send_command(self, command: str) -> bool:
        """Отправка команды в активный бэкенд."""
        with self._lock:
            if not self.running or not self.backend:
                self.error_callback(" Not connected")
                return False
        
        try:
            self.backend.send_command(command)
            return True
        except Exception as e:
            self.error_callback(f" Command failed: {e}")
            return False

    def _run_loop(self) -> None:
        """Запуск цикла чтения с обработкой ошибок."""
        try:
            if self.backend:
                self.backend.read_loop()
        except Exception as e:
            self.error_callback(f" Reader crashed: {e}")
        finally:
            with self._lock:
                self.running = False
            self.status_callback(" Reader stopped")

    def _cleanup(self) -> None:
        """Сброс состояния."""
        self.backend = None
        self.thread = None
        self.backend_type = None
        self.running = False

    def is_connected(self) -> bool:
        """Проверка состояния подключения."""
        with self._lock:
            return self.running and self.backend is not None