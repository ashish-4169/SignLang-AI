"""Fast unit tests for the non-camera parts. Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signlang.augment import augment, mirror  # noqa: E402
from signlang.config import DATA_PATH, MODEL_PATH  # noqa: E402
from signlang.landmarks import normalize_landmarks  # noqa: E402
from signlang.model import GestureClassifier  # noqa: E402
from signlang.sentence import PredictionSmoother, SentenceBuilder  # noqa: E402


def sample():
    return np.array(json.loads(DATA_PATH.read_text())[0]["landmarks"], dtype=np.float32)


def test_normalize_is_idempotent():
    x = sample()
    assert np.allclose(normalize_landmarks(x), x, atol=1e-5)


def test_augment_stays_normalized():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = augment(sample(), rng)
        assert a.shape == (63,)
        assert np.allclose(a[:3], 0)                       # wrist at origin
        assert abs(np.max(np.abs(a)) - 1) < 1e-4           # max-abs scaled


def test_mirror_twice_is_identity():
    x = sample()
    assert np.allclose(mirror(mirror(x)), x)


def test_classifier_predicts_known_sample():
    clf = GestureClassifier.load(MODEL_PATH)
    raw = json.loads(DATA_PATH.read_text())
    hits = sum(clf.predict(np.array(d["landmarks"]))[0] == d["label"] for d in raw[:200])
    assert hits > 180


def test_smoother_majority():
    s = PredictionSmoother(window=5, min_agreement=0.6)
    for l in ["A", "A", "B", "A", "A"]:
        out, _ = s.update(l, 0.9)
    assert out == "A"
    for l in ["B", "C", None]:
        out, _ = s.update(l, 0.9)
    assert out is None  # no clear majority


def test_builder_hold_commit_and_no_repeat():
    b = SentenceBuilder(hold_seconds=1.0, release_seconds=0.4)
    t = 0.0
    for _ in range(30):                      # hold "A" for 3 s
        b.update("A", t); t += 0.1
    assert b.text == "A"                     # added once, not 3 times
    b.update(None, t); t += 0.1              # one dropped frame: must not unlock
    for _ in range(15):
        b.update("A", t); t += 0.1
    assert b.text == "A"
    for _ in range(6):                       # hand gone for 0.6 s -> unlocked
        b.update(None, t); t += 0.1
    for _ in range(12):
        b.update("A", t); t += 0.1
    assert b.text == "AA"


def test_builder_words_letters_space_undo():
    b = SentenceBuilder()
    b.add_sign("Hello")
    b.add_sign("A"); b.add_sign("L")
    b.add_sign("Fist")                       # control gesture -> space
    b.add_sign("ILoveYou")
    assert b.text == "hello AL I love you "
    b.undo()
    assert b.text == "hello AL "
    b.backspace()
    assert b.text == "hello AL"
    b.clear()
    assert b.text == "" and b.history == []
