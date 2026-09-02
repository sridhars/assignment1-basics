import os
import pickle
import threading
import time

import psutil

from cs336_basics.train_bpe import train_bpe


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


def train_bpe_tinystories():
    """Run the train_bpe_tinystories deliverable: train on the full TinyStories
    dataset, vocab_size=10000, and serialize the result to disk."""
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    with PeakMemoryMonitor() as monitor:
        start = time.time()
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=16)
        elapsed = time.time() - start

    longest_token = max(vocab.values(), key=len)
    print(
        f"trained {len(vocab)}-token vocab in {elapsed / 60:.1f} min, "
        f"peak RAM {monitor.peak_bytes / 1024**3:.2f} GB"
    )
    print(f"longest token ({len(longest_token)} bytes): {longest_token!r}")

    out_path = "./data/tinystories_bpe.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print(f"saved vocab/merges to {out_path}")

    return vocab, merges


def main():
    train_bpe_tinystories()


if __name__ == "__main__":
    main()
