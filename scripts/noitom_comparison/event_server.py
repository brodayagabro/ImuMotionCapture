"""Generic, loopback-only JSON-lines event ingress. No experiment dependencies."""
import json
import socket
import socketserver
import threading
import time

MAX_MESSAGE_BYTES = 65536
PROTOCOL_VERSION = 1


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(.2)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buffer = bytearray()
        greeted = False
        try:
            while not self.server.stopping.is_set():
                try:
                    data = self.request.recv(8192)
                except socket.timeout:
                    continue
                if not data:
                    return
                buffer.extend(data)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    received_ns = time.monotonic_ns()
                    if len(line) > MAX_MESSAGE_BYTES:
                        raise ValueError("Message too large")
                    payload = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object")
                    kind = payload.get("type")
                    if kind == "hello":
                        if payload.get("protocol_version") != PROTOCOL_VERSION:
                            raise ValueError("Unsupported protocol version")
                        greeted = True
                    elif not greeted:
                        raise ValueError("Send hello first")
                    if kind in {"hello", "ping"}:
                        response = {"type": "ready", "protocol_version": PROTOCOL_VERSION,
                            **self.server.recorder.recording_state()}
                    else:
                        event_id = payload.get("event_id")
                        if not isinstance(kind, str) or not kind or len(kind) > 128:
                            raise ValueError("Invalid event type")
                        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
                            raise ValueError("Invalid event_id")
                        response = {"type": "ack", "event_id": event_id,
                            **self.server.recorder.external_event(payload, received_ns)}
                    self._send(response)
                if len(buffer) > MAX_MESSAGE_BYTES:
                    raise ValueError("Message too large")
        except (ValueError, UnicodeError) as error:
            try:
                self._send({"type": "error", "message": str(error)})
            except OSError:
                pass
        except OSError:
            pass  # peer closed or server stopped; acquisition is independent

    def _send(self, payload):
        self.request.sendall(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


class EventServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = False

    def __init__(self, recorder, port=5055):
        self.recorder = recorder
        self.stopping = threading.Event()
        super().__init__(("127.0.0.1", port), _Handler)
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": .05},
                                       name="external-event-server", daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server_address[1]

    def close(self):
        if not self.stopping.is_set():
            self.stopping.set()
            self.shutdown()
            self.server_close()
