# ================================================================
#  SignLang AI — Train GestureNet on Google Colab (Free GPU)
#  Paste this entire code into a Colab cell and press Shift+Enter
# ================================================================

# ── Step 1: Upload your gesture_data.json ────────────────────────
from google.colab import files
import json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

print("📂 Upload your gesture_data.json file...")
uploaded = files.upload()  # A button will appear — click it and upload gesture_data.json

# ── Step 2: Load data ─────────────────────────────────────────────
filename = list(uploaded.keys())[0]
with open(filename) as f:
    raw = json.load(f)

LABELS = sorted(list(set(d["label"] for d in raw)))
print(f"\n✅ Data loaded!")
print(f"   Total samples : {len(raw)}")
print(f"   Classes ({len(LABELS)}): {LABELS}")

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device        : {device}")
if device.type == "cuda":
    print(f"   GPU           : {torch.cuda.get_device_name(0)}")

# ── Step 3: Dataset ───────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(
            np.array([d["landmarks"] for d in data]), dtype=torch.float32)
        self.y = torch.tensor(
            [LABELS.index(d["label"]) for d in data], dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataset  = GestureDataset(raw)
n_val    = max(1, int(len(dataset) * 0.15))
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)
print(f"\n   Train: {n_train} | Val: {n_val}")

# ── Step 4: Model ─────────────────────────────────────────────────
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

model     = GestureNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 GestureNet ready | Parameters: {total_params:,}")

# ── Step 5: Training ──────────────────────────────────────────────
print("\n🚀 Training started...\n")
best_acc     = 0
train_losses = []
val_accs     = []

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

    val_acc  = correct / total
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
        }, "gesture_model.pth")

print(f"\n✅ Training complete!")
print(f"   Best Accuracy: {best_acc*100:.1f}%")

# ── Step 6: Evaluation ────────────────────────────────────────────
checkpoint = torch.load("gesture_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

all_preds, all_labels_list = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        preds = model(Xb).argmax(1)
        all_preds.extend(preds.numpy())
        all_labels_list.extend(yb.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels_list, all_preds, target_names=LABELS))

# ── Step 7: Plot training curves ──────────────────────────────────
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
plt.savefig('training_curves.png', dpi=150, facecolor='#0d1117')
plt.show()
print("\n📈 Training curves saved!")

# ── Step 8: Download model ────────────────────────────────────────
print("\n📥 Downloading gesture_model.pth to your laptop...")
files.download("gesture_model.pth")
print("✅ Download started! Save it to your signlang-ai/models/ folder")
