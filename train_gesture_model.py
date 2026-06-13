"""
train_gesture_model.py
======================
Trains a lightweight MLP on MediaPipe landmark data for custom gesture recognition.
This upgrades the rule-based classifier with a learned neural network.

Usage:
    1. Collect data:   python train_gesture_model.py --collect
    2. Train model:    python train_gesture_model.py --train
    3. Evaluate:       python train_gesture_model.py --eval

The trained model (gesture_model.pth) is auto-loaded by app.py.
"""

import os
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ── Gesture labels ─────────────────────────────────────────────────────────────
GESTURE_LABELS = [
    "Hello", "ThumbsUp", "Peace", "ILoveYou", "RockOn",
    "CallMe", "Fist", "Pointing", "A", "B", "C", "D",
    "I", "L", "K", "Three", "Four", "Five"
]
NUM_CLASSES = len(GESTURE_LABELS)
LANDMARK_DIM = 63  # 21 landmarks × 3 coords (x, y, z)

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7)

# ── Landmark extraction ────────────────────────────────────────────────────────
def extract_landmarks(frame):
    """Returns flattened (63,) landmark array or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark

    # Normalise: subtract wrist position, scale by max distance
    coords = np.array([[l.x, l.y, l.z] for l in lm])
    wrist = coords[0]
    coords -= wrist
    scale = np.max(np.abs(coords)) + 1e-8
    coords /= scale
    return coords.flatten().astype(np.float32)

# ── Neural Network ─────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    """Lightweight MLP for gesture classification from landmarks."""
    def __init__(self, input_dim=63, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ── Dataset ────────────────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data_path="data/gestures.json"):
        with open(data_path) as f:
            raw = json.load(f)
        self.X = torch.tensor(np.array([d["landmarks"] for d in raw]), dtype=torch.float32)
        self.y = torch.tensor([GESTURE_LABELS.index(d["label"]) for d in raw], dtype=torch.long)
        print(f"Dataset loaded: {len(self.X)} samples, {NUM_CLASSES} classes")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Data collection (interactive) ─────────────────────────────────────────────
def collect_data(samples_per_gesture=200):
    """Collect landmark data via webcam — press SPACE to record, Q to quit."""
    os.makedirs("data", exist_ok=True)
    data_path = "data/gestures.json"
    existing = json.load(open(data_path)) if os.path.exists(data_path) else []

    cap = cv2.VideoCapture(0)
    print("\n=== Data Collection Mode ===")
    print("Show each gesture to the camera and press SPACE to record.\n")

    for gesture in GESTURE_LABELS:
        count = 0
        print(f"▶  Show gesture: '{gesture}'  (target: {samples_per_gesture} samples)")
        print("   Press SPACE to record | Q to skip to next gesture")

        while count < samples_per_gesture:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            lm = extract_landmarks(frame)
            cv2.putText(frame, f"Gesture: {gesture} [{count}/{samples_per_gesture}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (79, 195, 247), 2)
            cv2.putText(frame, "SPACE=record  Q=skip",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (91, 143, 185), 1)

            if lm is not None:
                cv2.putText(frame, "Hand detected ✓", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (76, 175, 80), 1)

            cv2.imshow("Data Collection — SignLang AI", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and lm is not None:
                existing.append({"label": gesture, "landmarks": lm.tolist()})
                count += 1
                print(f"   Recorded {count}/{samples_per_gesture}", end="\r")
            elif key == ord('q'):
                break

        print(f"   Done: {count} samples collected for '{gesture}'")

    cap.release()
    cv2.destroyAllWindows()

    with open(data_path, "w") as f:
        json.dump(existing, f)
    print(f"\n✅ Dataset saved → {data_path} ({len(existing)} total samples)")

# ── Training ───────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = GestureDataset()
    n_val = max(1, int(len(dataset) * 0.15))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    model     = GestureNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc, train_losses, val_accs = 0, [], []

    print("\n Training GestureNet...")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(1)
                correct += (preds.cpu() == y_b).sum().item()
                total += len(y_b)
        val_acc = correct / total
        train_losses.append(total_loss / len(train_loader))
        val_accs.append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/gesture_model.pth")

    print(f"\n✅ Best Val Accuracy: {best_val_acc:.4f}")
    print("   Model saved → models/gesture_model.pth")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')
    ax1.plot(train_losses, color='#4fc3f7'); ax1.set_title('Training Loss', color='#e8f4fd')
    ax2.plot(val_accs, color='#81c784'); ax2.set_title('Validation Accuracy', color='#e8f4fd')
    plt.tight_layout()
    plt.savefig('static/training_curves.png', dpi=150, facecolor='#0d1117')
    print("   Curves saved → static/training_curves.png")

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate():
    dataset = GestureDataset()
    loader  = DataLoader(dataset, batch_size=128)
    model   = GestureNet()
    model.load_state_dict(torch.load("models/gesture_model.pth", map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_b.numpy())

    print(classification_report(all_labels, all_preds, target_names=GESTURE_LABELS))

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect training data via webcam")
    parser.add_argument("--train",   action="store_true", help="Train the gesture model")
    parser.add_argument("--eval",    action="store_true", help="Evaluate trained model")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    if args.collect: collect_data()
    elif args.train: train()
    elif args.eval:  evaluate()
    else:
        print("Usage: python train_gesture_model.py [--collect | --train | --eval]")
