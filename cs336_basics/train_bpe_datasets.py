import argparse
import pickle
import time
from typing import NamedTuple

from cs336_basics.memorymonitor import PeakMemoryMonitor
from cs336_basics.train_bpe import train_bpe


class Run(NamedTuple):
    input_path: str
    vocab_size: int
    out_path: str


# Deliverables: train_bpe_tinystories and train_bpe_expts_owt (<=12h, <=100 GB RAM).
RUNS = {
    "tinystories": Run("./data/TinyStoriesV2-GPT4-train.txt", 10000, "./data/tinystories_bpe.pkl"),
    "owt": Run("./data/owt_train.txt", 32000, "./data/owt_bpe.pkl"),
}


def run(name: str):
    """Train BPE on the named dataset and serialize vocab/merges to disk."""
    cfg = RUNS[name]
    special_tokens = ["<|endoftext|>"]

    with PeakMemoryMonitor() as monitor:
        start = time.time()
        vocab, merges = train_bpe(cfg.input_path, cfg.vocab_size, special_tokens, num_processes=16, show_progress=True)
        elapsed = time.time() - start

    longest_token = max(vocab.values(), key=len)
    print(
        f"trained {len(vocab)}-token vocab in {elapsed / 60:.1f} min, "
        f"peak RAM {monitor.peak_bytes / 1024**3:.2f} GB"
    )
    print(f"longest token ({len(longest_token)} bytes): {longest_token!r}")

    with open(cfg.out_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print(f"saved vocab/merges to {cfg.out_path}")

    return vocab, merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=RUNS)
    run(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
