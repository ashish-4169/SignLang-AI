"""
train.py — Train GestureNet MLP on collected landmark data
===========================================================
Run: python train.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH  = "data/gestures.json"
MODEL_PATH = "models/gesture_model.pth"
LABEL_PATH = "models/labels.json"

os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(DATA_PATH) as f:
    raw = json.load(f)

LABELS = sorted(list(set(d["label"] for d in raw)))
print(f"Classes ({len(LABELS)}): {LABELS}")
print(f"Total samples: {len(raw)}")

# Save labels
with open(LABEL_PATH, "w") as f:
    json.dump(LABELS, f)

# ── Dataset ───────────────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(
            np.array([d["landmarks"] for d in data]), dtype=torch.float32)
        self.y = torch.tensor(
            [LABELS.index(d["label"]) for d in data], dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataset = GestureDataset(raw)
n_val   = max(1, int(len(dataset) * 0.15))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)
print(f"Train: {n_train} | Val: {n_val}")

# ── Model ─────────────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=len(LABELS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining on: {device}")

model     = GestureNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc     = 0
train_losses = []
val_accs     = []

print("\nTraining GestureNet...\n")
for epoch in range(1, 101):
    # Train
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Validate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds = model(Xb.to(device)).argmax(1)
            correct += (preds.cpu() == yb).sum().item()
            total   += len(yb)

    val_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    val_accs.append(val_acc)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/100 | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "labels": LABELS,
            "input_dim": 63,
        }, MODEL_PATH)

print(f"\n✅ Best Accuracy: {best_acc*100:.1f}%")
print(f"   Model saved → {MODEL_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")["model_state"])
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        preds = model(Xb).argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

print("\n" + classification_report(all_labels, all_preds, target_names=LABELS))

# ── Save training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d1117')
for ax in [ax1, ax2]:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

ax1.plot(train_losses, color='#4fc3f7', lw=2)
ax1.set_title('Training Loss', color='#e8f4fd', pad=10)
ax1.set_xlabel('Epoch', color='#8b949e')

ax2.plot(val_accs, color='#81c784', lw=2)
ax2.axhline(y=best_acc, color='#ffeb3b', linestyle='--', alpha=0.7,
            label=f'Best: {best_acc*100:.1f}%')
ax2.set_title('Validation Accuracy', color='#e8f4fd', pad=10)
ax2.set_xlabel('Epoch', color='#8b949e')
ax2.legend(facecolor='#161b22', labelcolor='#e0f7fa')

plt.tight_layout()
plt.savefig('static/training_curves.png', dpi=150, facecolor='#0d1117')
print("\n📊 Training curves saved → static/training_curves.png")
print("\nNext step: streamlit run app.py")
