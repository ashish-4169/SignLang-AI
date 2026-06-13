"""
collect_data.py — Collect hand gesture training data via webcam
===============================================================
Run: python collect_data.py

Controls:
  SPACE = save current frame's landmarks
  N     = next gesture
  Q     = quit
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math

# ── Gestures to collect ───────────────────────────────────────────────────────
GESTURES = [
    "Hello", "ThumbsUp", "Fist", "Peace", "ILoveYou",
    "RockOn", "CallMe", "Pointing", "Three", "Four",
    "A", "B", "L", "I"
]
SAMPLES_PER_GESTURE = 150
DATA_FILE = "data/gesture_data.json"

os.makedirs("data", exist_ok=True)

# ── MediaPipe setup ───────────────────────────────────────────────────────────
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    USE_LEGACY = True
    print("Using MediaPipe legacy API")
except:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading model (~8MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.IMAGE
    )
    hands = mp_vision.HandLandmarker.create_from_options(options)
    USE_LEGACY = False
    print("Using MediaPipe Tasks API")

def extract_landmarks(frame_rgb):
    """Extract normalized 63-dim landmark vector from frame."""
    if USE_LEGACY:
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    else:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = hands.detect(mp_img)
        if not results.hand_landmarks:
            return None
        lm = results.hand_landmarks[0]
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)

    # Normalize: subtract wrist, scale by max distance
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.max(np.abs(coords)) + 1e-8
    coords /= scale
    return coords.flatten().tolist()

# ── Load existing data ────────────────────────────────────────────────────────
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} existing samples")
else:
    all_data = []

# ── Collection loop ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*50)
print("  GESTURE DATA COLLECTION")
print("="*50)
print("Controls: SPACE=save  N=next gesture  Q=quit")
print("="*50 + "\n")

for gesture_idx, gesture in enumerate(GESTURES):
    count = 0
    print(f"\n[{gesture_idx+1}/{len(GESTURES)}] Show gesture: '{gesture}'")
    print(f"  Need {SAMPLES_PER_GESTURE} samples. Press SPACE to save each one.")

    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm    = extract_landmarks(rgb)

        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (640,80), (13,31,60), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, f"Gesture: {gesture}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (79,195,247), 2)
        cv2.putText(frame, f"Saved: {count}/{SAMPLES_PER_GESTURE}",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (91,143,185), 1)
        cv2.putText(frame, "SPACE=save  N=skip  Q=quit",
                    (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (91,143,185), 1)

        if lm is not None:
            cv2.putText(frame, "Hand detected!", (15, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (76,175,80), 2)
            # Progress bar
            pct = int(count / SAMPLES_PER_GESTURE * 600)
            cv2.rectangle(frame, (15, 440), (615, 455), (30,50,80), -1)
            cv2.rectangle(frame, (15, 440), (15+pct, 455), (79,195,247), -1)
        else:
            cv2.putText(frame, "No hand detected", (15, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (244,67,54), 2)

        cv2.imshow("SignLang AI - Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and lm is not None:
            all_data.append({"label": gesture, "landmarks": lm})
            count += 1
            print(f"  {count}/{SAMPLES_PER_GESTURE}", end="\r")

        elif key == ord('n'):
            print(f"  Skipped (saved {count})")
            break

        elif key == ord('q'):
            print("\nQuitting early...")
            cap.release()
            cv2.destroyAllWindows()
            with open(DATA_FILE, "w") as f:
                json.dump(all_data, f)
            print(f"Saved {len(all_data)} samples to {DATA_FILE}")
            exit()

    print(f"  Done! {count} samples saved for '{gesture}'")

cap.release()
cv2.destroyAllWindows()

with open(DATA_FILE, "w") as f:
    json.dump(all_data, f)

print(f"\n{'='*50}")
print(f"  Collection complete!")
print(f"  Total samples: {len(all_data)}")
print(f"  Saved to: {DATA_FILE}")
print(f"{'='*50}")
print("\nNow run: python train.py")
