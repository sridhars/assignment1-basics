import os
import threading

import psutil


class PeakMemoryMonitor:
    """Tracks peak combined RSS of this process plus its live children."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
