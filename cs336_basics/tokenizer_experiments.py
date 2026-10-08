"""2.7 tokenizer_experiments — one-off to fill answers.md."""

import pickle
import time
from functools import cache
from itertools import accumulate
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
DATA = Path("data")


def load(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return Tokenizer(blob["vocab"], blob["merges"], special_tokens=[EOT])


@cache
def _worker_tok(path):
    tok = load(path)
    tok._merge_bytes = cache(tok._merge_bytes)
    return tok


def sample(path, n=10):
    with open(path, encoding="utf-8", errors="replace") as f:
        chunk = next(s for s in accumulate(f) if s.count(EOT) >= n)
    return EOT.join(chunk.split(EOT)[:n]) + EOT


def ratio(tok, text):
    return (b := len(text.encode()), t := len(tok.encode(text)), b / t)


def _encode_range(args):
    path, start, end, tok_path = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    return np.asarray(_worker_tok(tok_path).encode(text), dtype=np.uint16)


def encode_dataset(tok_path, in_path, out_name):
    out = DATA / out_name
    if out.exists():
        ids = np.load(out, mmap_mode="r")
        print(f"  {out.name}: {ids.size:,} toks, max={int(ids.max())}, {in_path.stat().st_size / ids.size:.3f} B/tok")
        return
    with open(in_path, "rb") as f:
        bounds = find_chunk_boundaries(f, max(in_path.stat().st_size // (1 << 20), 1), EOT.encode())
    ranges = [(str(in_path), a, b, str(tok_path)) for a, b in zip(bounds[:-1], bounds[1:])]
    with Pool(16) as pool:
        parts = list(tqdm(pool.imap(_encode_range, ranges, chunksize=4), total=len(ranges), desc=out.name))
    ids = np.concatenate(parts)
    np.save(out, ids)
    print(f"  wrote {out.name}: {ids.size:,} toks, max={int(ids.max())}")


if __name__ == "__main__":
    ts, owt = load(DATA / "tinystories_bpe.pkl"), load(DATA / "owt_bpe.pkl")
    ts_s, owt_s = sample(DATA / "TinyStoriesV2-GPT4-train.txt"), sample(DATA / "owt_train.txt")
    print("(a) ts", ratio(ts, ts_s), "owt", ratio(owt, owt_s))
    print("(b) ts-on-owt", ratio(ts, owt_s))
    print("   ", [ts.vocab[i].decode("utf-8", "replace") for i in ts.encode(owt_s[:400])])

    def bps(tok, path):
        text = open(path, "rb").read(8_000_000).decode("utf-8", "replace")
        t0 = time.perf_counter()
        tok.encode(text)
        return len(text.encode()) / (time.perf_counter() - t0)

    ts_bps = bps(ts, DATA / "TinyStoriesV2-GPT4-valid.txt")
    owt_bps = bps(owt, DATA / "owt_valid.txt")
    print(f"(c) {ts_bps / 1e6:.2f} / {owt_bps / 1e6:.2f} MB/s; pile {825e9 / min(ts_bps, owt_bps) / 86400:.1f} days")

    print("(d) uint16 (10k/32k vocab < 65535; uint8 overflows, uint32 doubles disk)")
    encode_dataset(DATA / "tinystories_bpe.pkl", DATA / "TinyStoriesV2-GPT4-valid.txt", "tinystories_valid_ids.npy")
    encode_dataset(DATA / "tinystories_bpe.pkl", DATA / "TinyStoriesV2-GPT4-train.txt", "tinystories_train_ids.npy")
    encode_dataset(DATA / "owt_bpe.pkl", DATA / "owt_valid.txt", "owt_valid_ids.npy")
    encode_dataset(DATA / "owt_bpe.pkl", DATA / "owt_train.txt", "owt_train_ids.npy")
