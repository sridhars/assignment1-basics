import pickle
import time

from cs336_basics.memorymonitor import PeakMemoryMonitor
from cs336_basics.train_bpe import train_bpe


def train_bpe_expts_owt():
    """Run the train_bpe_expts_owt deliverable: train on the full OpenWebText
    dataset, vocab_size=32000, and serialize the result to disk.

    Resource requirements per the assignment: <=12 hours, <=100 GB RAM.
    """
    input_path = "./data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    with PeakMemoryMonitor() as monitor:
        start = time.time()
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=16, show_progress=True)
        elapsed = time.time() - start

    longest_token = max(vocab.values(), key=len)
    print(
        f"trained {len(vocab)}-token vocab in {elapsed / 60:.1f} min, "
        f"peak RAM {monitor.peak_bytes / 1024**3:.2f} GB"
    )
    print(f"longest token ({len(longest_token)} bytes): {longest_token!r}")

    out_path = "./data/owt_bpe.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print(f"saved vocab/merges to {out_path}")

    return vocab, merges


def main():
    train_bpe_expts_owt()


if __name__ == "__main__":
    main()
