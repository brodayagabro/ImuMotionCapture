"""Install the checksum-pinned MediaMTX receiver inside this server directory."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request


def install(root):
    root = Path(root)
    spec = json.loads((root / "requirements-tools.json").read_text())["mediamtx"]
    arch = {"x86_64": "linux_amd64", "aarch64": "linux_arm64", "armv7l": "linux_armv7"}.get(platform.machine())
    if arch not in spec["sha256"]:
        raise ValueError(f"Unsupported MediaMTX architecture: {platform.machine()}")
    tools = root / ".tools"
    (tools / "downloads").mkdir(parents=True, exist_ok=True)
    (tools / "bin").mkdir(exist_ok=True)
    name = f'mediamtx_v{spec["version"]}_{arch}.tar.gz'
    archive = tools / "downloads" / name

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if not archive.exists() or sha(archive) != spec["sha256"][arch]:
        temporary = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(spec["base_url"] + name, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha(temporary) != spec["sha256"][arch]:
                raise ValueError("MediaMTX checksum mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    binary = tools / "bin" / "mediamtx"
    with tarfile.open(archive) as bundle:
        member = bundle.getmember("mediamtx")
        if not member.isfile():
            raise ValueError("MediaMTX archive member is not a regular file")
        data = bundle.extractfile(member).read()
        # Do not replace a running, already correct installation on every setup.
        if not binary.exists() or hashlib.sha256(binary.read_bytes()).digest() != hashlib.sha256(data).digest():
            temporary = binary.with_suffix(".new")
            temporary.write_bytes(data)
            temporary.chmod(0o755)
            temporary.replace(binary)
    subprocess.run([str(binary), "--version"], check=True)
    return binary


if __name__ == "__main__":
    install(Path(__file__).resolve().parent)
