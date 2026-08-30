from __future__ import annotations


class SentenceBuffer:
    """Accumulates streamed LLM text and yields finished sentences.

    A sentence is complete when its punctuation (. ! ?) is followed by a space,
    so short replies are emitted as soon as they can be read out loud. A hard
    character cap keeps latency bounded for run-on un-punctuated output."""

    def __init__(self, min_chars: int = 3, max_chars: int = 160):
        self._min = min_chars
        self._max = max_chars
        self._buf = ""
        self._start = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sent = self._buf[self._start:cut].strip()
            self._start = self._advance(cut)
            if sent:
                out.append(sent)
        return out

    def flush(self) -> str | None:
        rem = self._buf[self._start:].strip()
        self._buf = ""
        self._start = 0
        return rem or None

    # -- internals ---------------------------------------------------------

    def _find_cut(self) -> int | None:
        b, start = self._buf, self._start
        tail_len = len(b) - start
        for i in range(len(b) - 1, start - 1, -1):
            ch = b[i]
            if ch not in ".!?":
                continue
            j = i + 1
            while j < len(b) and b[j] in "\"')\u201d\u2019]":
                j += 1
            if j >= len(b):
                continue
            if not b[j].isspace():
                continue
            if len(b[start:i]) >= self._min:
                return j
        if tail_len >= self._max:
            comma = b.rfind(", ", start, len(b))
            if comma > start:
                return comma + 2
            space = b.rfind(" ", start, start + self._max)
            if space > start:
                return space + 1
            return min(start + self._max, len(b))
        return None

    def _advance(self, cut: int) -> int:
        b = self._buf
        k = cut
        while k < len(b) and b[k].isspace():
            k += 1
        return k
