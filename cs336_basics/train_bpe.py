import itertools
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce

import regex as re

from cs336_basics.pretokenization import find_chunk_boundaries


def train_bpe_chunk(start: int, end: int, input_path, special_token_list: list[str]):
    special_escape_tokens = [re.escape(x) for x in special_token_list]
    with open(input_path, "rb") as f:
        f.seek(start)
        big_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sub_chunks = re.split("|".join(special_escape_tokens), big_chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counter: Counter[tuple[bytes, ...]] = Counter()
        for chunk in sub_chunks:
            encoded_chunks = (x.group().encode("utf-8") for x in re.finditer(PAT, chunk))
            counter.update(tuple([x[i : i + 1] for i in range(len(x))]) for x in encoded_chunks)
        return counter


def process_word(
    orig_word: tuple[bytes, ...],
    merge_pair: tuple[bytes, bytes],
    pairs_count,
    pairs_to_words: defaultdict[tuple[bytes, bytes], list[tuple[bytes, ...]]],
    freq: int,
):
    merged_pair = b"".join(merge_pair)
    to_subtract: list[tuple[bytes, bytes]] = []
    to_add: list[tuple[bytes, bytes]] = []
    new_word: tuple[bytes, ...] = ()
    i = 0
    while i < len(orig_word):
        pair = None if i == len(orig_word) - 1 else (orig_word[i], orig_word[i + 1])
        if i == len(orig_word) - 1 or pair != merge_pair:
            new_word += (orig_word[i],)
            i += 1
        else:
            to_subtract.append(merge_pair)
            if new_word:
                to_subtract.append((new_word[-1], orig_word[i]))
                to_add.append((new_word[-1], merged_pair))
            if i + 2 < len(orig_word):
                to_subtract.append((orig_word[i + 1], orig_word[i + 2]))
                to_add.append((merged_pair, orig_word[i + 2]))
            new_word += (merged_pair,)
            i += 2

    for i in range(len(orig_word) - 1):
        pair = (orig_word[i], orig_word[i + 1])
        if pair in pairs_to_words and orig_word in pairs_to_words[pair]:
            pairs_to_words[pair].remove(orig_word)
            if len(pairs_to_words[pair]) == 0:
                del pairs_to_words[pair]

    for i in range(len(new_word) - 1):
        pair = (new_word[i], new_word[i + 1])
        if new_word not in pairs_to_words[pair]:
            pairs_to_words[pair].append(new_word)
    for pair in to_add:
        pairs_count[pair] += freq

    for pair in to_subtract:
        pairs_count[pair] -= freq
        if pairs_count[pair] == 0:
            del pairs_count[pair]
    return new_word


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4):
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    eot_string = "<|endoftext|>"
    assert eot_string in special_tokens
    eot_token = eot_string.encode("utf-8")
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, eot_token)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(train_bpe_chunk, start, end, input_path, special_tokens)
                for start, end in itertools.pairwise(boundaries)
            ]
            counters = [future.result() for future in as_completed(futures)]
            pretoken_counts = reduce(lambda x, y: x + y, counters)
    pairs_to_words: defaultdict[tuple[bytes, bytes], list[tuple[bytes, ...]]] = defaultdict(list)
    pairs_count: Counter[tuple[bytes, bytes]] = Counter()

    for word, count in pretoken_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if word not in pairs_to_words[pair]:
                pairs_to_words[pair].append(word)
            pairs_count[pair] += count

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and len(pairs_count) > 0:
        max_count = max(pairs_count.values())
        candidate_pairs = [k for k, v in pairs_count.items() if v == max_count]
        merge_pair = max(candidate_pairs)
        merged_pair = b"".join(merge_pair)
        merges.append(merge_pair)
        vocab[len(vocab)] = merged_pair
        assert merge_pair in pairs_to_words
        matching_words = pairs_to_words[merge_pair][:]
        for orig_word in matching_words:
            new_word = process_word(
                orig_word,
                merge_pair,
                pairs_count,
                pairs_to_words,
                pretoken_counts[orig_word],
            )
            assert len(new_word) < len(orig_word)
            pretoken_counts[new_word] += pretoken_counts[orig_word]
            del pretoken_counts[orig_word]
    return vocab, merges


def main():
    eot_string = "<|endoftext|>"
    _vocab, _merges = train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 1000, [eot_string])
    print(len(_vocab), len(_merges))


if __name__ == "__main__":
    main()
