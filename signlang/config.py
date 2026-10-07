"""
Central configuration: file paths and how each gesture turns into text.
Every script imports from here, so paths only need to change in one place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "gesture_data.json"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_model.pth"
METRICS_PATH = MODEL_DIR / "metrics.json"
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
STATIC_DIR = ROOT / "static"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Gestures the data collector knows about (order = collection order).
GESTURES = [
    "Hello", "ThumbsUp", "Fist", "Peace", "ILoveYou",
    "RockOn", "CallMe", "Pointing", "Three", "Four",
    "A", "B", "L", "I",
]

# What each gesture adds to the sentence builder.
#   - single characters (letters, digits) are appended directly, so letters spell words
#   - longer strings are added as whole words, separated by spaces
#   - SPACE is a control gesture that ends the current word
# Edit this to change the vocabulary; any label not listed is added as-is.
SPACE = "<SPACE>"
GESTURE_TEXT = {
    "Hello": "hello",
    "ThumbsUp": "good",
    "Peace": "peace",
    "ILoveYou": "I love you",
    "RockOn": "rock on",
    "CallMe": "call me",
    "Pointing": "you",
    "Three": "3",
    "Four": "4",
    "Fist": SPACE,
    "A": "A",
    "B": "B",
    "L": "L",
    "I": "I",
}
