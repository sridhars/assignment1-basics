from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import reduce
from typing import BinaryIO
import os
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe_chunk(start: int, end: int, input_path, special_escape_tokens: list[str]):

    with open(input_path, "rb") as f:
        f.seek(start)
        big_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sub_chunks = re.split("|".join(special_escape_tokens), big_chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counter = Counter()
        for chunk in sub_chunks:
            encoded_chunks = map(lambda x: x.group().encode("utf-8"), re.finditer(PAT, chunk))
            counter.update(map(lambda x: tuple([x[i : i + 1] for i in range(len(x))]), encoded_chunks))
        return counter


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab: dict[int, bytes] = { i:bytes([i]) for i in range(256)}
    eot_string = "<|endoftext|>"
    eot_token = eot_string.encode('utf-8')
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, eot_token)
        special_escape_tokens = list(map(lambda x: re.escape(x), special_tokens + [eot_string]))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(train_bpe_chunk, start, end, input_path, special_escape_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            counters = [future.result() for future in as_completed(futures)]

            pretoken_counts = reduce(lambda x, y: x + y, counters)
            print(pretoken_counts.most_common(5))
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        print(len(vocab))
        pairs_count = Counter()
        for token_tuple, count in pretoken_counts.items():
            for i in range(len(token_tuple) - 1):
                pairs_count[(token_tuple[i], token_tuple[i + 1])] += count
        max_count = max(pairs_count.values())
        candidate_pairs = [k for k, v in pairs_count.items() if v == max_count]
        merge_pair = max(candidate_pairs)
        merges.append(merge_pair)
        vocab[len(vocab)] = b''.join(merge_pair)

        keys = list(pretoken_counts.keys())
        for token_tuple in keys:
            for i in range(len(token_tuple) - 1):
                if (token_tuple[i], token_tuple[i+1]) == merge_pair:
                    pretoken_counts[token_tuple[:i] + (token_tuple[i] + token_tuple[i+1],) + token_tuple[i+1:]] = pretoken_counts[token_tuple]
                    if token_tuple in pretoken_counts:
                        del pretoken_counts[token_tuple]
    return vocab, merges


def main():
    train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 10000, [])


if __name__ == "__main__":
    main()
