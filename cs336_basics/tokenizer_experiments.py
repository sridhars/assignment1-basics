"""Experiments with trained BPE tokenizers (assignment problem tokenizer_experiments / 2.7)."""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_BYTES = SPECIAL_TOKEN.encode("utf-8")

DATA_DIR = Path("./data")
TS_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TS_VALID = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"
TS_TOKENIZER_PATH = DATA_DIR / "tinystories_bpe.pkl"
OWT_TOKENIZER_PATH = DATA_DIR / "owt_bpe.pkl"

# Assignment states the Pile is 825GB of text.
PILE_BYTES = 825 * 10**9
NUM_PROCESSES = 16
CHUNK_BYTES = 1 << 20
THROUGHPUT_BYTES = 8_000_000

_WORKER_TOKENIZER: Tokenizer | None = None


def load_tokenizer(path: str | Path) -> Tokenizer:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return Tokenizer(blob["vocab"], blob["merges"], special_tokens=[SPECIAL_TOKEN])


def sample_documents(path: Path, n: int = 10) -> list[str]:
    """Return the first `n` complete documents, each including the trailing special token."""
    docs: list[str] = []
    leftover = ""
    with path.open(encoding="utf-8", errors="replace") as f:
        while len(docs) < n:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            leftover += chunk
            parts = leftover.split(SPECIAL_TOKEN)
            leftover = parts[-1]
            for part in parts[:-1]:
                doc = part + SPECIAL_TOKEN
                if doc.strip() not in ("", SPECIAL_TOKEN):
                    docs.append(doc)
                    if len(docs) >= n:
                        return docs[:n]
    if leftover.strip() and len(docs) < n:
        docs.append(leftover)
    return docs[:n]


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def compression_ratio(tokenizer: Tokenizer, text: str) -> tuple[int, int, float]:
    n_bytes = utf8_bytes(text)
    n_tokens = len(tokenizer.encode(text))
    return n_bytes, n_tokens, n_bytes / n_tokens if n_tokens else float("nan")


def sample_token_strings(tokenizer: Tokenizer, text: str, limit: int = 40) -> list[str]:
    ids = tokenizer.encode(text)
    pieces = []
    for token_id in ids[:limit]:
        pieces.append(tokenizer.vocab[token_id].decode("utf-8", errors="replace"))
    return pieces


def run_compression_experiments(ts_tok: Tokenizer, owt_tok: Tokenizer) -> dict:
    ts_docs = sample_documents(TS_TRAIN, 10)
    owt_docs = sample_documents(OWT_TRAIN, 10)
    ts_sample = "".join(ts_docs)
    owt_sample = "".join(owt_docs)

    ts_on_ts = compression_ratio(ts_tok, ts_sample)
    owt_on_owt = compression_ratio(owt_tok, owt_sample)
    ts_on_owt = compression_ratio(ts_tok, owt_sample)
    owt_on_ts = compression_ratio(owt_tok, ts_sample)

    print("sampled documents")
    print(f"  TinyStories: {len(ts_docs)} docs, {ts_on_ts[0]} utf-8 bytes")
    print(f"  OpenWebText: {len(owt_docs)} docs, {owt_on_owt[0]} utf-8 bytes")
    print()
    print("(a) matched compression ratio (bytes/token)")
    print(f"  TinyStories tokenizer on TinyStories: {ts_on_ts[1]} tokens, {ts_on_ts[2]:.3f} bytes/token")
    print(f"  OpenWebText tokenizer on OpenWebText: {owt_on_owt[1]} tokens, {owt_on_owt[2]:.3f} bytes/token")
    print()
    print("(b) cross-tokenizer compression")
    print(f"  TinyStories tokenizer on OpenWebText: {ts_on_owt[1]} tokens, {ts_on_owt[2]:.3f} bytes/token")
    print(f"  OpenWebText tokenizer on TinyStories: {owt_on_ts[1]} tokens, {owt_on_ts[2]:.3f} bytes/token")

    snippet = owt_docs[0][:400]
    print()
    print("  OpenWebText snippet tokenized by TinyStories tokenizer:")
    print("   ", sample_token_strings(ts_tok, snippet, 50))
    print("  same snippet tokenized by OpenWebText tokenizer:")
    print("   ", sample_token_strings(owt_tok, snippet, 50))

    return {
        "ts_on_ts": ts_on_ts,
        "owt_on_owt": owt_on_owt,
        "ts_on_owt": ts_on_owt,
        "owt_on_ts": owt_on_ts,
        "ts_sample_bytes": ts_on_ts[0],
        "owt_sample_bytes": owt_on_owt[0],
    }


def measure_throughput(tokenizer: Tokenizer, path: Path) -> tuple[int, int, float, float]:
    """Encode a prefix of `path` and return (utf8_bytes, n_tokens, seconds, bytes_per_sec)."""
    with path.open("rb") as f:
        raw = f.read(THROUGHPUT_BYTES)
    text = raw.decode("utf-8", errors="replace")
    n_bytes = utf8_bytes(text)
    start = time.perf_counter()
    n_tokens = len(tokenizer.encode(text))
    elapsed = time.perf_counter() - start
    bps = n_bytes / elapsed if elapsed else float("inf")
    return n_bytes, n_tokens, elapsed, bps


def run_throughput_experiments(ts_tok: Tokenizer, owt_tok: Tokenizer) -> dict:
    ts_n_bytes, ts_n_tokens, ts_s, ts_bps = measure_throughput(ts_tok, TS_VALID)
    owt_n_bytes, owt_n_tokens, owt_s, owt_bps = measure_throughput(owt_tok, OWT_VALID)
    # Use the slower of the two as a conservative Pile estimate.
    conservative_bps = min(ts_bps, owt_bps)
    pile_seconds = PILE_BYTES / conservative_bps
    pile_hours = pile_seconds / 3600
    pile_days = pile_hours / 24

    print()
    print("(c) tokenizer throughput")
    print(
        f"  TinyStories tokenizer: {ts_n_bytes} bytes / {ts_n_tokens} tokens in {ts_s:.2f}s "
        f"-> {ts_bps:,.0f} bytes/s ({ts_bps / 1e6:.2f} MB/s)"
    )
    print(
        f"  OpenWebText tokenizer: {owt_n_bytes} bytes / {owt_n_tokens} tokens in {owt_s:.2f}s "
        f"-> {owt_bps:,.0f} bytes/s ({owt_bps / 1e6:.2f} MB/s)"
    )
    print(
        f"  Pile (825GB) at min throughput {conservative_bps:,.0f} bytes/s: "
        f"{pile_seconds:,.0f}s ({pile_hours:.1f} hours, {pile_days:.2f} days)"
    )
    return {
        "ts_bps": ts_bps,
        "owt_bps": owt_bps,
        "conservative_bps": conservative_bps,
        "pile_seconds": pile_seconds,
        "pile_days": pile_days,
    }


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    tokenizer = load_tokenizer(tokenizer_path)
    orig = tokenizer._merge_bytes
    cache: dict[bytes, list[bytes]] = {}

    def cached_merge(byte_chunk: bytes) -> list[bytes]:
        hit = cache.get(byte_chunk)
        if hit is None:
            hit = orig(byte_chunk)
            cache[byte_chunk] = hit
        return hit

    tokenizer._merge_bytes = cached_merge
    _WORKER_TOKENIZER = tokenizer


def chunk_ranges(path: Path) -> list[tuple[int, int]]:
    """~1 MiB slices, snapped forward to a newline or <|endoftext|>."""
    size = path.stat().st_size
    if size == 0:
        return []
    splits = (b"\n", SPECIAL_TOKEN_BYTES)
    ranges: list[tuple[int, int]] = []
    start = 0
    with path.open("rb") as f:
        while start < size:
            guess = min(start + CHUNK_BYTES, size)
            if guess == size:
                ranges.append((start, size))
                break
            f.seek(guess)
            pos = guess
            end = size
            while pos < size:
                mini = f.read(4096)
                if not mini:
                    break
                hits = [(mini.find(tok), len(tok)) for tok in splits]
                hits = [(i, ntok) for i, ntok in hits if i != -1]
                if hits:
                    i, ntok = min(hits)
                    end = pos + i + ntok
                    break
                pos += len(mini)
            ranges.append((start, end))
            start = end
    return ranges


def _encode_chunk(args: tuple[str, int, int]) -> np.ndarray:
    input_path, start, end = args
    assert _WORKER_TOKENIZER is not None
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    ids = _WORKER_TOKENIZER.encode(text)
    return np.asarray(ids, dtype=np.uint16)


def encode_dataset(tokenizer_path: Path, input_path: Path, output_path: Path) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        tokens = np.load(output_path, mmap_mode="r")
        print(
            f"  skip existing {output_path} "
            f"({tokens.size:,} tokens, max_id={int(tokens.max()) if tokens.size else 'n/a'})"
        )
        return tokens

    ranges = [(str(input_path), start, end) for start, end in chunk_ranges(input_path)]
    print(f"  {input_path.name}: {len(ranges)} chunks of ~{CHUNK_BYTES / 1024**2:.2f} MiB")

    parts: list[np.ndarray | None] = [None] * len(ranges)
    with ProcessPoolExecutor(
        max_workers=NUM_PROCESSES,
        initializer=_init_worker,
        initargs=(str(tokenizer_path),),
    ) as pool:
        futures = {pool.submit(_encode_chunk, r): i for i, r in enumerate(ranges)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=output_path.name):
            parts[futures[fut]] = fut.result()

    tokens = np.concatenate(parts) if parts else np.array([], dtype=np.uint16)
    if tokens.size:
        max_id = int(tokens.max())
        if max_id > np.iinfo(np.uint16).max:
            raise OverflowError(f"token id {max_id} does not fit in uint16")
    np.save(output_path, tokens)
    n_bytes = input_path.stat().st_size
    ratio = n_bytes / tokens.size if tokens.size else float("nan")
    print(
        f"  wrote {output_path} ({tokens.size:,} tokens, max_id={int(tokens.max()) if tokens.size else 'n/a'}, "
        f"{ratio:.3f} bytes/token, {output_path.stat().st_size / 1024**2:.1f} MiB on disk)"
    )
    return tokens


def run_dataset_encoding() -> None:
    jobs = [
        (TS_TOKENIZER_PATH, TS_VALID, DATA_DIR / "tinystories_valid_ids.npy"),
        (TS_TOKENIZER_PATH, TS_TRAIN, DATA_DIR / "tinystories_train_ids.npy"),
        (OWT_TOKENIZER_PATH, OWT_VALID, DATA_DIR / "owt_valid_ids.npy"),
        (OWT_TOKENIZER_PATH, OWT_TRAIN, DATA_DIR / "owt_train_ids.npy"),
    ]
    print()
    print("(d) encode train/valid datasets as uint16")
    print("  uint16 max is 65535; TinyStories vocab is 10_000 and OpenWebText is 32_000, so every id fits,")
    print("  while uint8 (max 255) cannot and uint32 would double the on-disk size.")
    for tokenizer_path, input_path, output_path in jobs:
        encode_dataset(tokenizer_path, input_path, output_path)


def main() -> None:
    ts_tok = load_tokenizer(TS_TOKENIZER_PATH)
    owt_tok = load_tokenizer(OWT_TOKENIZER_PATH)
    print(f"loaded TinyStories tokenizer vocab={len(ts_tok.vocab)} from {TS_TOKENIZER_PATH}")
    print(f"loaded OpenWebText tokenizer vocab={len(owt_tok.vocab)} from {OWT_TOKENIZER_PATH}")
    run_compression_experiments(ts_tok, owt_tok)
    run_throughput_experiments(ts_tok, owt_tok)
    run_dataset_encoding()


if __name__ == "__main__":
    main()
