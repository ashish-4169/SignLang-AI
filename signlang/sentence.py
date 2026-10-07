"""
Turning a noisy per-frame prediction stream into text.

PredictionSmoother  - majority vote over the last N frames, so the label doesn't flicker.
SentenceBuilder     - commits a sign to the sentence once it has been held steady
                      for `hold_seconds`. To add the same sign twice in a row, change
                      to another sign or lower your hand briefly.
"""

import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .config import GESTURE_TEXT, SPACE


class PredictionSmoother:
    def __init__(self, window: int = 8, min_agreement: float = 0.6):
        self.window = window
        self.min_agreement = min_agreement
        self._buf: deque = deque(maxlen=window)

    def update(self, label: str | None, confidence: float = 0.0) -> tuple[str | None, float]:
        """Add one frame's prediction (None = no hand). Returns (stable_label, mean_confidence)."""
        self._buf.append((label, confidence))
        votes = Counter(l for l, _ in self._buf)
        top, n = votes.most_common(1)[0]
        if top is None or n / len(self._buf) < self.min_agreement:
            return None, 0.0
        conf = sum(c for l, c in self._buf if l == top) / n
        return top, conf

    def reset(self):
        self._buf.clear()


@dataclass
class SentenceBuilder:
    hold_seconds: float = 1.0
    release_seconds: float = 0.4                   # hand must be gone this long to allow a repeat
    text: str = ""
    history: list = field(default_factory=list)   # committed chunks, for undo
    _candidate: str | None = None
    _since: float = 0.0
    _locked: str | None = None                    # last committed sign, must change before repeat
    _none_since: float | None = None

    def progress(self, now: float | None = None) -> float:
        """0..1 — how far the current sign is towards being committed (for a progress bar)."""
        if self._candidate is None or self._candidate == self._locked:
            return 0.0
        now = time.monotonic() if now is None else now
        return min(1.0, (now - self._since) / max(self.hold_seconds, 1e-6))

    def update(self, label: str | None, now: float | None = None) -> str | None:
        """Feed the stable label for this frame. Returns the label if it was just committed."""
        now = time.monotonic() if now is None else now
        if label is None:
            # Only treat "no sign" as real once it lasts `release_seconds`;
            # a single dropped frame shouldn't unlock or reset anything.
            if self._none_since is None:
                self._none_since = now
            if now - self._none_since >= self.release_seconds:
                self._candidate, self._locked = None, None
            return None
        self._none_since = None
        if label != self._candidate:
            self._candidate, self._since = label, now
            return None
        if label == self._locked:
            return None
        if now - self._since >= self.hold_seconds:
            self.add_sign(label)
            self._locked = label
            return label
        return None

    # ── editing ──────────────────────────────────────────────────────────────
    def add_sign(self, label: str):
        token = GESTURE_TEXT.get(label, label)
        if token == SPACE:
            self.add_space()
        elif len(token) == 1:              # letter / digit: spell into current word
            self._append(token)
        else:                              # whole word / phrase
            chunk = token + " "
            if self.text and not self.text.endswith(" "):
                chunk = " " + chunk
            self._append(chunk)

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self._append(" ")

    def undo(self):
        if self.history:
            chunk = self.history.pop()
            self.text = self.text[: len(self.text) - len(chunk)]

    def backspace(self):
        if not self.text:
            return
        self.text = self.text[:-1]
        if self.history:
            last = self.history.pop()[:-1]
            if last:
                self.history.append(last)

    def clear(self):
        self.text = ""
        self.history.clear()

    def _append(self, chunk: str):
        self.text += chunk
        self.history.append(chunk)
