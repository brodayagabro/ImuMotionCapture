"""Asynchronous JSON-lines client. ACKs never pace the stimulus clock."""
import ipaddress
import json
import time
import uuid

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtNetwork import QAbstractSocket, QTcpSocket

MAX_MESSAGE_BYTES = 65536


class RecorderClient(QObject):
    readiness_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    acknowledged = pyqtSignal(object)
    failed = pyqtSignal(str)
    controlled = pyqtSignal(object)

    def __init__(self, parent=None, *, timeout_ms=3000, continuous_control=False):
        super().__init__(parent)
        self.continuous_control = continuous_control
        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self._connected)
        self.socket.readyRead.connect(self._read)
        self.socket.disconnected.connect(self._disconnected)
        self.socket.errorOccurred.connect(lambda _: self._fail(self.socket.errorString()))
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._check)
        self.timeout_ns = timeout_ms * 1_000_000
        self.ready = False
        self.recording_session_id = None
        self.pending = {}
        self.control_pending = {}
        self.control_supported = False
        self.control_state = {}
        self.buffer = bytearray()
        self.last_response_ns = self.last_ping_ns = 0
        self._failing = False

    def connect_recorder(self, host="127.0.0.1", port=5055):
        if host.lower() == "localhost":
            host = "127.0.0.1"
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local or not 1 <= port <= 65535:
            raise ValueError("Нужен localhost и порт от 1 до 65535")
        self.close()
        self.buffer.clear()
        self.recording_session_id = None
        self.last_response_ns = self.last_ping_ns = time.monotonic_ns()
        self.status_changed.emit("CONNECTING · подключение к рекордеру…")
        self.timer.start()
        self.socket.connectToHost(host, port)

    def _connected(self):
        self.socket.setSocketOption(QAbstractSocket.SocketOption.LowDelayOption, 1)
        self._write({"type": "hello", "protocol_version": 1})

    def _set_ready(self, ready):
        if self.ready != ready:
            self.ready = ready
            self.readiness_changed.emit(ready)

    def _write(self, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(data) > MAX_MESSAGE_BYTES or self.socket.state() != QAbstractSocket.SocketState.ConnectedState:
            return False
        return self.socket.write(data) == len(data)

    def send_event(self, payload):
        if not self.ready:
            return False
        if len(self.pending) >= 128:
            self._fail("Слишком много событий без подтверждения")
            return False
        event_id = payload["event_id"]
        if payload.get("recording_session_id") != self.recording_session_id:
            return False
        self.pending[event_id] = time.monotonic_ns()
        if not self._write(payload):
            self._fail("Не удалось отправить событие")
            return False
        return True

    def send_control(self, command, **payload):
        request_id = uuid.uuid4().hex
        if not self.control_supported or self.socket.state() != QAbstractSocket.SocketState.ConnectedState:
            raise ValueError('Подключите рекордер с поддержкой управления записью')
        message = {'type': 'control', 'request_id': request_id, 'command': command, **payload}
        self.control_pending[request_id] = time.monotonic_ns()
        if not self._write(message):
            self.control_pending.pop(request_id, None)
            raise ValueError('Не удалось отправить команду рекордеру')
        return request_id

    def _read(self):
        self.buffer.extend(bytes(self.socket.readAll()))
        try:
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if len(line) > MAX_MESSAGE_BYTES:
                    raise ValueError("Слишком длинный ответ рекордера")
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("Неверный ответ рекордера")
                self.last_response_ns = time.monotonic_ns()
                if message.get("type") == "ready":
                    self.control_supported = bool(message.get('control_supported', False))
                    if message.get("protocol_version") != 1 or type(message.get("ready")) is not bool:
                        raise ValueError("Несовместимый протокол рекордера")
                    session_id = message.get("recording_session_id")
                    if message["ready"] and (not isinstance(session_id, str) or not session_id):
                        raise ValueError("Рекордер не сообщил ID записи")
                    if self.ready and session_id != self.recording_session_id and not self.continuous_control:
                        self._fail("Запись рекордера остановлена или сменилась; подключитесь снова")
                        return
                    self.recording_session_id = session_id
                    self._set_ready(message["ready"])
                    self.status_changed.emit("CONNECTED · READY · REC" if self.ready else
                        ("CONNECTED · рекордер подключён" if self.continuous_control else
                         "CONNECTED · включите «REC · эксперимент» в рекордере"))
                elif message.get('type') == 'control_result':
                    if message.get('request_id') not in self.control_pending:
                        raise ValueError('Неизвестный ответ управления')
                    del self.control_pending[message['request_id']]
                    if message.get('accepted'):
                        self.control_state = message
                        if 'ready' in message:
                            self.recording_session_id = message.get('recording_session_id')
                            self._set_ready(bool(message['ready']))
                    self.controlled.emit(message)
                elif message.get("type") == "ack":
                    if message.get("event_id") not in self.pending:
                        raise ValueError("Неизвестное подтверждение события")
                    if message.get("accepted") is not True:
                        self._fail("Рекордер отклонил событие: " + str(message.get("reason")))
                        return
                    if (message.get("recording_session_id") != self.recording_session_id or
                            type(message.get("host_timestamp_ns")) is not int):
                        raise ValueError("Неверная временная метка или ID записи в ACK")
                    del self.pending[message["event_id"]]
                    self.acknowledged.emit(message)
                else:
                    raise ValueError(str(message.get("message", "Неверный ответ рекордера")))
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise ValueError("Слишком длинный ответ рекордера")
        except (ValueError, UnicodeError) as error:
            self._fail(str(error))

    def _check(self):
        now = time.monotonic_ns()
        if now - self.last_response_ns > self.timeout_ns or any(now - t > self.timeout_ns for t in (*self.pending.values(), *self.control_pending.values())):
            self._fail("Нет ответа или подтверждения события от рекордера")
        elif now - self.last_ping_ns >= 1_000_000_000:
            self.last_ping_ns = now
            self._write({"type": "ping"})

    def _disconnected(self):
        if not self._failing:
            self._fail("Связь с рекордером потеряна")

    def _fail(self, message):
        if self._failing:
            return
        self._failing = True
        self.timer.stop()
        self._set_ready(False)
        self.pending.clear()
        self.control_pending.clear()
        self.control_supported = False
        self.buffer.clear()
        self.socket.abort()
        self.status_changed.emit("DISCONNECTED · " + message)
        self.failed.emit(message)
        self._failing = False

    def close(self):
        self._failing = True
        self.timer.stop()
        self._set_ready(False)
        self.socket.abort()
        self.pending.clear()
        self.control_pending.clear()
        self.control_supported = False
        self._failing = False
