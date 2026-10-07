"""
Hand detection with MediaPipe plus landmark normalization.

HandDetector hides the difference between MediaPipe's legacy `solutions` API
and the newer Tasks API, so the rest of the code just gets a list of hands.
"""

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .config import HAND_LANDMARKER_PATH, HAND_LANDMARKER_URL

# The 21-point hand skeleton (same as mp.solutions.hands.HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize_landmarks(coords) -> np.ndarray:
    """(21, 3) raw landmarks -> 63-dim vector: wrist at origin, scaled to [-1, 1]."""
    c = np.asarray(coords, dtype=np.float32).reshape(21, 3).copy()
    c -= c[0]
    c /= np.max(np.abs(c)) + 1e-8
    return c.flatten()


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3) in image-relative coords (x, y in 0..1)
    handedness: str        # "Left" / "Right"

    @property
    def features(self) -> np.ndarray:
        return normalize_landmarks(self.landmarks)


class HandDetector:
    def __init__(self, max_hands: int = 1, static_image: bool = False,
                 min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        self._mp = mp
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static_image, max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
            self.api = "legacy"
        except AttributeError:
            # Newer MediaPipe releases dropped `solutions`; fall back to the Tasks API.
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            if not os.path.exists(HAND_LANDMARKER_PATH):
                HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=mp_vision.RunningMode.IMAGE)
            self._hands = mp_vision.HandLandmarker.create_from_options(opts)
            self.api = "tasks"

    def detect(self, img_rgb: np.ndarray) -> list[Hand]:
        hands = []
        if self.api == "legacy":
            res = self._hands.process(img_rgb)
            for hl, hi in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
                coords = np.array([[l.x, l.y, l.z] for l in hl.landmark], dtype=np.float32)
                hands.append(Hand(coords, hi.classification[0].label))
        else:
            res = self._hands.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img_rgb))
            for i, hl in enumerate(res.hand_landmarks or []):
                coords = np.array([[l.x, l.y, l.z] for l in hl], dtype=np.float32)
                side = res.handedness[i][0].category_name if res.handedness else "Right"
                hands.append(Hand(coords, side))
        return hands

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass


def draw_hand(img: np.ndarray, hand: Hand, label: str | None = None,
              line_color=(91, 91, 214), point_color=(34, 197, 94)) -> None:
    """Draw the skeleton (and optional label) on an RGB image in place."""
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], line_color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 5, point_color, -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        x, y = pts[0]
        cv2.putText(img, label, (x, max(20, y - 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, line_color, 2, cv2.LINE_AA)
