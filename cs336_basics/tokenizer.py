from collections.abc import Iterable, Iterator
import pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._special_set = set(self.special_tokens) if self.special_tokens else ()
        vocab_set = set(vocab.values())
        for st in self._special_set:
            if st.encode("utf-8") not in vocab_set:
                self.vocab[len(vocab)] = st.encode("utf-8")
        self._special_re = re.compile("({})".format("|".join([re.escape(x) for x in sorted(self._special_set, key=len, reverse=True)])))
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.bytes_to_id = {v:k for k, v in vocab.items()}


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, 'rb') as vocab_f:
            vocab = pickle.load(vocab_f)
        with open(merges_filepath, 'rb') as merges_f:
            merges = pickle.load(merges_f)
        return cls(vocab, merges, special_tokens)

    def _merge_bytes(self, byte_chunk:bytes) -> list[bytes]:
        parts = [bytes([b]) for b in byte_chunk]
        while len(parts) > 1:
            best = min(
                (
                (self.merge_ranks[pair], i) for i in range(len(parts) - 1)
                if (pair:= (parts[i], parts[i + 1])) in self.merge_ranks
                ),
                default = None
            )
            if not best:
                break
            
            i = best[1]
            parts[i: i + 2] = [parts[i] + parts[i+1]]
        return parts


    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in PAT_RE.finditer(text):
            ids += [self.bytes_to_id[x] for x in self._merge_bytes(match.group().encode('utf-8'))]
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if not text:
            return ids
        parts = self._special_re.split(text) if self.special_tokens else [text]
        for part in parts:
            if part in self._special_set:
                ids += [self.bytes_to_id[part.encode("utf-8")]]
            else:
                ids += self._encode_ordinary(part)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            yield from self.encode(item)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[x] for x in ids]).decode("utf-8", errors="replace")
