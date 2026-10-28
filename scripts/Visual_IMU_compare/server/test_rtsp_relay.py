import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rtsp_relay", Path(__file__).with_name("rtsp_relay.py"))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def test_relay_config_ipv6_and_no_extra_listeners():
    args = relay.arguments(["--bind", "::1", "--port", "18554"])
    text = relay.relay_config(args.bind, args.port)
    assert '"[::1]:18554"' in text
    assert "overridePublisher: false" in text
    assert "hls: no" in text and "webrtc: no" in text and "moq: no" in text


@pytest.mark.parametrize("argv", [["--port", "0"], ["--duration", "nan"], ["--bind", "localhost:8080"]])
def test_invalid_relay_arguments(argv):
    with pytest.raises(SystemExit):
        relay.arguments(argv)
