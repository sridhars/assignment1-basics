# Written answers

## 2.1 (unicode1)

(a) `chr(0)` returns the null character U+0000, whose string value is `'\x00'`.

(b) `__repr__()` shows the escaped form `'\x00'`, while printing the character emits a non-printable NUL with no visible glyph.

(c) Python can store NUL inside a string, but printing often looks truncated or empty because the NUL is not rendered; the bytes of the string still contain `'\x00'`.

## 2.2 (unicode2)

(a) UTF-8 uses fewer bytes than UTF-16 or UTF-32 on ASCII-heavy English text (one byte per ASCII character instead of two or four) and is the encoding already used by nearly all web text.

(b) Example: `"helloこんに".encode("utf-8")`. The function decodes each byte as its own UTF-8 string, but Japanese characters are 3-byte sequences, so continuation bytes such as `\x81` are not valid UTF-8 on their own.

(c) `[0xFF, 0xFF]`. Byte `0xFF` is never a valid UTF-8 lead or continuation byte, so this two-byte sequence does not decode to any Unicode character.

## 2.4 (train_bpe_tinystories)

(a) Training a byte-level BPE tokenizer (`vocab_size=10000`) on the full TinyStories train set took 1.9 minutes and 7.63 GB peak RAM (`num_processes=16`), well under the 30 min / 30 GB budget. The longest tokens are a 3-way tie at 15 bytes: `b' accomplishment'`, `b' disappointment'`, and `b' responsibility'`, which makes sense since TinyStories are GPT-4-generated children's stories that reuse a small set of moral/lesson words often enough to fully merge them.

(b) TODO: profile with cProfile/py-spy and report the bottleneck.

## 2.4 (train_bpe_expts_owt)

(a) Training the 32,000-token OpenWebText BPE tokenizer took 113.7 minutes and 63.81 GB peak RAM. The longest token is 64 bytes of repeated over-encoded UTF-8 (`b'\xc3\x83\xc3\x82...'` and long runs of `---` / `===`), which is expected in scraped web text with encoding artifacts and visual separators.

(b) TinyStories merges whole children's-story words (`accomplishment`, `veterinarian`, `strawberries`), while OpenWebText's larger vocab includes web tokens (`http`, `https`, `html`), punctuation runs, and adult/news words (`cryptocurrencies`, `Charlottesville`, `telecommunications`).

## 2.7 (tokenizer_experiments)

(a) On 10 TinyStories train documents the TinyStories tokenizer (10K vocab) compresses at 4.161 bytes/token (7565 bytes / 1818 tokens). On 10 OpenWebText train documents the OpenWebText tokenizer (32K vocab) compresses at 4.704 bytes/token (31617 bytes / 6722 tokens).

(b) Tokenizing the same OpenWebText sample with the TinyStories tokenizer drops compression to 3.199 bytes/token (9883 tokens). The TinyStories vocab over-splits adult/web words that never occurred in children's stories (e.g. `Calling` → `C`/`all`/`ing`, `urban` → `ur`/`b`/`an`, `supernatural` → `super`/`n`/`atur`/`al`).

(c) Encode throughput is about 1.16 MB/s for the TinyStories tokenizer and 1.00 MB/s for the OpenWebText tokenizer. At the slower rate, tokenizing the Pile (825GB) would take about 822,461 seconds ≈ 9.5 days.

(d) `uint16` is appropriate because both vocabularies fit in 16 bits (observed max ids 9,999 and 31,999; `uint16` max is 65,535), whereas `uint8` overflows at 255 and `uint32` would double the on-disk size of the tokenized datasets. Serialized `data/{tinystories,owt}_{train,valid}_ids.npy` as `uint16` (TinyStories train 541,229,347 tokens / 4.116 B/tok; OpenWebText train 2,727,120,452 tokens / 4.371 B/tok).
