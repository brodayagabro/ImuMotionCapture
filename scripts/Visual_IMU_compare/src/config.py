"""Receiver options; no camera inference or native suit SDK on this computer."""
from dataclasses import asdict, dataclass
import ipaddress
from .paths import RECORDS_ROOT


@dataclass(frozen=True)
class ComparisonConfig:
    bind: str = '0.0.0.0'
    port: int = 7001
    transport: str = 'udp'
    source_ip: str | None = None
    stale_timeout_ms: int = 500
    recordings_dir: str = str(RECORDS_ROOT)
    event_port: int = 0

    def __post_init__(self):
        # Explicit IPv4 LAN addresses, matching the documented deployment.
        ipaddress.IPv4Address(self.bind)
        if self.source_ip is not None:
            ipaddress.IPv4Address(self.source_ip)
        if self.transport not in {'udp', 'tcp'}:
            raise ValueError('transport must be udp or tcp')
        if not 1 <= self.port <= 65535 or not 1 <= self.stale_timeout_ms <= 60000:
            raise ValueError('Invalid receiver port or stale timeout')
        if not 0 <= self.event_port <= 65535:
            raise ValueError('Invalid event port')

    def metadata(self):
        return asdict(self)
