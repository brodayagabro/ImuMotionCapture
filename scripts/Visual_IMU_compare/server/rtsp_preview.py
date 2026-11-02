"""One bounded, shared JPEG preview per RTSP camera, decoded on the server."""
import asyncio
import time

from rtsp_recording import input_options, terminate


class Preview:
    def __init__(self, url, ffmpeg="ffmpeg", timeout=10):
        self.url, self.ffmpeg, self.timeout = url, ffmpeg, timeout
        self.jpeg = None
        self.updated = self.requested = 0
        self.task = None
        self.process = None
        self.closed = False
        self.error = "Подключение к видеопотоку…"

    def snapshot(self):
        self.requested = time.monotonic()
        if not self.closed and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self.run())
        if self.jpeg and time.monotonic() - self.updated <= 2:
            return self.jpeg
        return None

    async def run(self):
        while not self.closed and time.monotonic() - self.requested < 15:
            self.jpeg = None
            pending = bytearray()
            try:
                options = await input_options(self.ffmpeg, self.timeout)
                self.process = await asyncio.create_subprocess_exec(
                    self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                    *options, "-i", self.url, "-map", "0:v:0", "-an", "-filter_threads", "1",
                    "-vf", "fps=5,scale=w='min(iw,640)':h=-2", "-c:v", "mjpeg",
                    "-threads:v", "1", "-q:v", "5", "-f", "image2pipe", "pipe:1",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                while not self.closed and time.monotonic() - self.requested < 15:
                    block = await asyncio.wait_for(self.process.stdout.read(65536), self.timeout + 2)
                    if not block:
                        raise RuntimeError("Видеопоток недоступен; повторное подключение…")
                    pending.extend(block)
                    while pending:
                        start = pending.find(b"\xff\xd8")
                        if start < 0:
                            pending[:] = pending[-1:]
                            break
                        if start:
                            del pending[:start]
                        end = pending.find(b"\xff\xd9", 2)
                        if end < 0:
                            if len(pending) > 8 * 1024 * 1024:
                                pending.clear()
                            break
                        self.jpeg, self.updated = bytes(pending[:end + 2]), time.monotonic()
                        self.error = None
                        del pending[:end + 2]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = str(exc) or "Нет свежих кадров; повторное подключение…"
            finally:
                self.jpeg = None
                if self.process:
                    await terminate(self.process)
            if not self.closed:
                await asyncio.sleep(1)

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
