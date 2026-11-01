import json
import socket
import time
import pytest
from scripts.Visual_IMU_compare.src.config import ComparisonConfig
from scripts.Visual_IMU_compare.src.visual_client import VisualClient


def free_port(transport='udp'):
    with socket.socket(type=socket.SOCK_DGRAM if transport=='udp' else socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def receive(client, timeout=2):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        frames = client.poll_events()
        if frames: return frames[0]
        time.sleep(.002)
    raise AssertionError('No BVH frame received')


@pytest.mark.parametrize('transport', ['udp','tcp'])
def test_network_delivery_split_tcp_and_reconnect(packet_pair, transport):
    port = free_port(transport)
    client = VisualClient(ComparisonConfig(bind='127.0.0.1',port=port,transport=transport,source_ip='127.0.0.1'))
    client.connect(); client.start()
    kind = socket.SOCK_DGRAM if transport=='udp' else socket.SOCK_STREAM
    source = socket.socket(type=kind)
    try:
        source.connect(('127.0.0.1',port))
        header = json.dumps(packet_pair['skeleton']).encode()
        frame = json.dumps(packet_pair['frame']).encode()
        if transport=='udp':
            source.send(header); source.send(frame)
        else:
            source.sendall(header[:30])
            assert client.poll_events()==[]
            source.sendall(header[30:]+b'\n'+frame+b'\n')
        received = receive(client)
        assert received.frame_index==packet_pair['frame']['frame_id']
        assert time.perf_counter_ns()-received.host_timestamp_ns < 500_000_000
        assert client.poll_events()==[]
        client.stop()
        assert client.poll_events()==[]
        client.start()
        if transport=='tcp':
            source.close(); source=socket.socket(); source.connect(('127.0.0.1',port))
            source.sendall(header+b'\n'+frame+b'\n')
        else:
            source.send(header); source.send(frame)
        assert receive(client).frame_index==received.frame_index
    finally:
        source.close(); client.disconnect()


def test_source_filter_and_claim_release_on_bad_packet(packet_pair):
    port=free_port()
    client=VisualClient(ComparisonConfig(bind='127.0.0.1',port=port,source_ip='192.0.2.1'))
    client.connect(); client.start()
    with socket.socket(type=socket.SOCK_DGRAM) as source:
        source.sendto(json.dumps(packet_pair['skeleton']).encode(),('127.0.0.1',port))
        assert client.poll_events()==[] and client.decoder.metadata is None
    client.disconnect()
    client=VisualClient(ComparisonConfig(bind='127.0.0.1',port=port))
    client.connect(); client.start()
    released=[]
    client.stamp=lambda: (123,'claim')
    client.release_claim=lambda claim,error=None: released.append((claim,error)) if claim is not None else None
    try:
        with socket.socket(type=socket.SOCK_DGRAM) as source:
            source.sendto(b'{bad JSON',('127.0.0.1',port))
            with pytest.raises(ValueError): client.poll_events()
        assert len(released)==1 and released[0][0]=='claim' and released[0][1]
    finally: client.disconnect()
