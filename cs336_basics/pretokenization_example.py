from concurrent.futures import ProcessPoolExecutor
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


## Usage
# with open("../data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
with open("./data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    eot_token = b"<|endoftext|>"
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, eot_token)
    special_tokens = [eot_token]
    special_escape_tokens = list(map(lambda x: re.escape(x.decode("utf-8")), special_tokens))

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    pretoken_counts_list = list(Counter() for i in range(len(boundaries) - 1))
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        f.seek(start)
        big_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sub_chunks = re.split("|".join(special_escape_tokens), big_chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in sub_chunks:
            encoded_chunks = map(lambda x: x.group().encode("utf-8"), re.finditer(PAT, chunk))
            pretoken_counts_list[index].update(
                map(lambda x: tuple([x[i : i + 1] for i in range(len(x))]), encoded_chunks)
            )
    pretoken_counts = reduce(lambda x, y: x + y, pretoken_counts_list)
    print(pretoken_counts.most_common(5))
