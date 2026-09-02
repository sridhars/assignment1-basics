unicode 1:
(a) '\x00'

(b) its a nested string: "'\x00'"

(c) when printed, its not rendered. but in the bytes of the string, it is stored as '\x00'

unicode 2:
(a) utf 8 uses fewer bytes for the same string 

(b) "helloこんに"

(c) [255, 255] or [0xFF, 0xFF]

train_bpe_tinystories:
(a) Training a byte-level BPE tokenizer (vocab_size=10000) on the full TinyStories train set took 1.9 minutes and 7.63 GB peak RAM (num_processes=16), well under the 30 min / 30 GB budget. The longest tokens are a 3-way tie at 15 bytes: b' accomplishment', b' disappointment', and b' responsibility', which makes sense since TinyStories are GPT-4-generated stories that lean heavily on a small set of moral/lesson vocabulary, so these words recur often enough to fully merge into single tokens.

(b) TODO: profile with cProfile/py-spy and report the bottleneck.

