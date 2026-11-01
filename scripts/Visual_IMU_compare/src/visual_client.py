"""BVH listener and acquisition worker. Both UDP and TCP are server-push inputs."""
from dataclasses import replace
import queue
import socket
import threading
import time
from pyqt_mocap.mocap_core import SEGMENT_NAMES
from .visual_adapter import VisualAdapter
from .visual_protocol import BvhDecoder, MAX_MESSAGE_BYTES, PROTOCOL


class VisualClient:
    version = PROTOCOL

    def __init__(self, config):
        self.config = config
        self.delivery_mode = config.transport + '_bvh_listener'
        self.socket = self.peer = None
        self.running = False
        self.decoder = BvhDecoder()
        self.buffer = bytearray()
        self.udp_peer = None
        self.last_packet_s = 0.
        self.stamp = lambda: (time.perf_counter_ns(), None)
        self.release_claim = lambda claim, error=None: None
        self.discontinuity = lambda reason: None

    def connect(self):
        if self.socket is not None:
            return
        kind = socket.SOCK_DGRAM if self.config.transport == 'udp' else socket.SOCK_STREAM
        sock = socket.socket(socket.AF_INET, kind)
        try:
            if kind == socket.SOCK_STREAM:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.config.bind, self.config.port))
            if kind == socket.SOCK_STREAM:
                sock.listen(2)
            sock.setblocking(False)
            self.socket = sock
        except Exception:
            sock.close()
            raise

    def start(self):
        if self.socket is None:
            raise ValueError('Connect the visual listener first')
        if self.running:
            return
        self.decoder = BvhDecoder()
        self.buffer.clear()
        self.udp_peer = None
        if self.config.transport == 'udp':
            # Discard packets queued while the receiver was stopped.
            for _ in range(512):
                try:
                    self.socket.recvfrom(MAX_MESSAGE_BYTES + 1)
                except BlockingIOError:
                    break
            else:
                self.disconnect(); self.connect()
        self.running = True

    def stop(self):
        self.running = False
        self._close_peer()
        self.decoder = BvhDecoder()

    def disconnect(self):
        self.stop()
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _close_peer(self):
        if self.peer is not None:
            self.peer.close()
            self.peer = None
        self.buffer.clear()

    def _message(self):
        if self.config.transport == 'udp':
            try:
                payload, peer = self.socket.recvfrom(MAX_MESSAGE_BYTES + 1)
            except BlockingIOError:
                return None
            if self.config.source_ip and peer[0] != self.config.source_ip:
                return None
            if (self.udp_peer is not None and peer != self.udp_peer
                    and time.monotonic() - self.last_packet_s < self.config.stale_timeout_ms / 1000):
                return None
            return payload, peer
        if self.peer is None:
            try:
                peer, address = self.socket.accept()
            except BlockingIOError:
                return None
            if self.config.source_ip and address[0] != self.config.source_ip:
                peer.close()
                return None
            peer.setblocking(False)
            self.peer, self.decoder = peer, BvhDecoder()
        if b'\n' not in self.buffer:
            try:
                data = self.peer.recv(8192)
            except BlockingIOError:
                return None
            except (ConnectionResetError, ConnectionAbortedError):
                data = b''
            if not data:
                self._close_peer()
                self.discontinuity('TCP connection closed')
                return None
            self.buffer.extend(data)
        if b'\n' not in self.buffer:
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise ValueError('BVH TCP message too large')
            return None
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return bytes(line), None

    def poll_events(self):
        if not self.running:
            return []
        for _ in range(32):
            item = self._message()
            if item is None:
                return []
            payload, peer = item
            received_ns, claim = self.stamp()
            try:
                previous = self.decoder.metadata
                frame = self.decoder.decode(payload, received_ns)
                metadata = self.decoder.metadata
                if previous and metadata and any(previous[k] != metadata[k] for k in ('stream_id', 'skeleton_id')):
                    self.discontinuity('Visual stream or skeleton changed')
                if metadata is not None:
                    self.udp_peer, self.last_packet_s = peer, time.monotonic()
                if frame is not None:
                    result = replace(frame, capture_context=claim)
                    claim = None  # Ownership passes to recorder.visual_frame().
                    return [result]
            except Exception as error:
                self.release_claim(claim, str(error))
                claim = None
                raise
            finally:
                self.release_claim(claim)
        return []


class VisualReceiverThread(threading.Thread):
    def __init__(self, client, recorder, enabled_segments=SEGMENT_NAMES):
        super().__init__(name='visual-bvh-receiver', daemon=True)
        self.client, self.recorder = client, recorder
        client.stamp = recorder.stamp_visual_packet
        client.release_claim = recorder.release_claim
        client.discontinuity = lambda reason: recorder.event('visual_discontinuity', {'reason': reason}, incomplete=True)
        self.commands, self.events = queue.Queue(), queue.Queue(maxsize=256)
        self.capture_lock = threading.RLock()
        self.adapter = VisualAdapter()
        self.shutdown = threading.Event()
        self.streaming = False
        self.heading_offset_rad = 0.
        self.set_enabled_segments(enabled_segments)

    def set_enabled_segments(self, enabled):
        enabled = frozenset(enabled)
        if enabled.difference(SEGMENT_NAMES):
            raise ValueError('Unknown visual segment')
        self.enabled_segments = enabled

    def command(self, name):
        if name not in {'connect', 'start', 'stop', 'disconnect', 'shutdown'}:
            raise ValueError('Unknown receiver command')
        self.commands.put(name)

    def _emit(self, event):
        # GUI backlog is bounded; recording already consumed every input frame.
        while True:
            try:
                self.events.put_nowait(event)
                return
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass

    def _consume(self, frame):
        enabled, heading = self.enabled_segments, self.heading_offset_rad
        try:
            pose = self.adapter.orientations(frame, enabled, heading_offset_rad=heading)
            # Dataset readiness requires all five segments regardless of visibility.
            self.adapter.orientations(frame, heading_offset_rad=heading)
            error = None
        except (ValueError, KeyError) as exc:
            pose, error = {}, str(exc)
        self.recorder.visual_frame(frame, pose, error, enabled_segments=enabled, heading_offset_rad=heading)
        self._emit(('frame', frame, (pose, error)))

    def run(self):
        try:
            while not self.shutdown.is_set():
                try:
                    command = self.commands.get(timeout=0 if self.streaming else .05)
                except queue.Empty:
                    command = None
                try:
                    if command == 'shutdown':
                        break
                    if command:
                        getattr(self.client, command)()
                        if command in {'stop', 'disconnect'}:
                            self.streaming = False
                            self.recorder.event('visual_' + command, incomplete=True)
                        elif command == 'connect':
                            self.heading_offset_rad = 0.
                        elif command == 'start':
                            self.streaming = True
                        state = {'connect': 'CONNECTED', 'start': 'STREAMING',
                                 'stop': 'CONNECTED', 'disconnect': 'DISCONNECTED'}[command]
                        self._emit(('state', state, ''))
                    if self.streaming:
                        frames = self.client.poll_events()
                        for frame in frames:
                            self._consume(frame)
                        if not frames:
                            self.shutdown.wait(.002)
                except Exception as error:
                    self.streaming = False
                    self.recorder.event('visual_error', {'message': str(error)}, incomplete=True)
                    self.client.disconnect()
                    self._emit(('state', 'ERROR', str(error)))
        finally:
            self.streaming = False
            self.client.disconnect()
            self.shutdown.set()
