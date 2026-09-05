"""Heap + index-linked-list BPE merge sketch. Not wired into Tokenizer."""

import heapq


def _merge_bytes(self, chunk: bytes) -> list[bytes]:
    n, ranks = len(chunk), self.merge_ranks
    tok = [bytes([b]) for b in chunk]
    prev, nxt, dead = list(range(-1, n - 1)), [*range(1, n), -1], [False] * n

    def triple(i):
        j = nxt[i] if i >= 0 else -1
        r = ranks.get((tok[i], tok[j])) if j >= 0 else None
        return None if r is None else (r, i, j)

    heap = [t for i, _ in enumerate(tok[1:]) if (t := triple(i))]
    heapq.heapify(heap)
    while heap:
        _, i, j = heapq.heappop(heap)
        if dead[i] or dead[j] or nxt[i] != j:
            continue
        tok[i], dead[j], nxt[i] = tok[i] + tok[j], True, nxt[j]
        if nxt[i] >= 0:
            prev[nxt[i]] = i
        for t in filter(None, (triple(prev[i]), triple(i))):
            heapq.heappush(heap, t)
    return [t for t, d in zip(tok, dead) if not d]
