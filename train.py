"""
train.py — Train GestureNet on collected landmark data
======================================================
    python train.py                      # defaults
    python train.py --epochs 200
    python train.py --no-augment --no-class-weights   # baseline, for comparison

What it does
  * stratified 70 / 15 / 15 train / val / test split (every class in every split)
  * class-weighted loss, so rare classes (e.g. 'I') aren't ignored
  * on-the-fly augmentation: mirroring (left hands), rotation, stretch, jitter
  * keeps the checkpoint with the best validation macro-F1
  * reports per-class precision/recall/F1 on the held-out test set,
    plus accuracy on mirrored test samples (left-hand check)
  * writes models/gesture_model.pth, models/metrics.json,
    static/training_curves.png and static/confusion_matrix.png

Works the same locally or in Colab (`!python train.py`).
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlang.augment import augment, mirror  # noqa: E402
from signlang.config import DATA_PATH, METRICS_PATH, MODEL_PATH, STATIC_DIR  # noqa: E402
from signlang.model import INPUT_DIM, GestureNet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--out", default=str(MODEL_PATH))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--aug-strength", type=float, default=0.4,
                   help="scale for rotation/stretch/jitter (0 = mirroring only)")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LandmarkDataset(Dataset):
    def __init__(self, X, y, do_augment=False, seed=0, strength=1.0):
        self.X, self.y = X.astype(np.float32), y.astype(np.int64)
        self.do_augment = do_augment
        self.kw = dict(max_roll_deg=15.0 * strength, max_tilt_deg=10.0 * strength,
                       stretch=0.1 * strength, jitter=0.015 * strength)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = augment(self.X[i], self.rng, **self.kw) if self.do_augment else self.X[i]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(self.y[i])


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X.astype(np.float32))).argmax(1).numpy()


def stratified_split(X, y, seed):
    """70/15/15, keeping every class in every split where possible."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    # Classes with a single sample left in the temp split can't be stratified again.
    counts = Counter(y_tmp)
    strat = y_tmp if min(counts.values()) >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=strat, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_plots(history, cm, labels, best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STATIC_DIR.mkdir(exist_ok=True)
    ink, muted, grid = "#1a1f36", "#6b7280", "#e8eaf2"
    purple, green = "#5b5bd6", "#16a34a"

    def style(ax, title):
        ax.set_title(title, color=ink, fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ep = range(1, len(history["loss"]) + 1)
    a1.plot(ep, history["loss"], color=purple, lw=2)
    style(a1, "Training loss")
    a1.set_xlabel("Epoch", color=muted)
    a2.plot(ep, history["val_acc"], color=purple, lw=2, label="Val accuracy")
    a2.plot(ep, history["val_f1"], color=green, lw=2, label="Val macro-F1")
    a2.axvline(best_epoch, color=muted, ls="--", lw=1)
    a2.set_ylim(0, 1.02)
    style(a2, "Validation")
    a2.set_xlabel("Epoch", color=muted)
    a2.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "training_curves.png", dpi=150, facecolor="white")
    plt.close(fig)

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8, color=ink)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=ink)
    ax.set_xlabel("Predicted", color=muted)
    ax.set_ylabel("True", color=muted)
    ax.set_title("Test-set confusion matrix (row-normalized)", color=ink, fontsize=11,
                 fontweight="bold", loc="left")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else ink)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "confusion_matrix.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)

    raw = json.loads(Path(args.data).read_text())
    labels = sorted({d["label"] for d in raw})
    X = np.array([d["landmarks"] for d in raw], dtype=np.float32)
    y = np.array([labels.index(d["label"]) for d in raw])
    counts = Counter(d["label"] for d in raw)

    print(f"Loaded {len(raw)} samples, {len(labels)} classes")
    for lbl in labels:
        flag = "  <- low, collect more" if counts[lbl] < 50 else ""
        print(f"  {lbl:<10} {counts[lbl]:>4}{flag}")

    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y, args.seed)
    print(f"\nSplit: train {len(y_tr)} | val {len(y_val)} | test {len(y_te)}")

    train_loader = DataLoader(
        LandmarkDataset(X_tr, y_tr, do_augment=not args.no_augment, seed=args.seed,
                        strength=args.aug_strength),
        batch_size=args.batch_size, shuffle=True, drop_last=len(y_tr) % args.batch_size == 1)

    model = GestureNet(INPUT_DIM, len(labels))
    if args.no_class_weights:
        weights = None
    else:
        # Inverse-frequency weights, normalized so the average weight is 1.
        freq = np.bincount(y_tr, minlength=len(labels)).astype(np.float32)
        w = 1.0 / np.maximum(freq, 1)
        weights = torch.tensor(w / w.mean(), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"loss": [], "val_acc": [], "val_f1": []}
    best = (-1.0, -1.0)  # (macro-F1, accuracy)
    best_state, best_epoch = None, 0

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        pv = predict(model, X_val)
        acc = accuracy_score(y_val, pv)
        f1 = f1_score(y_val, pv, average="macro", zero_division=0)
        history["loss"].append(total / len(train_loader))
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        if (f1, acc) > best:
            best = (f1, acc)
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{args.epochs}  loss {history['loss'][-1]:.4f}  "
                  f"val acc {acc*100:5.1f}%  val macro-F1 {f1*100:5.1f}%")

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "labels": labels, "input_dim": INPUT_DIM,
                "normalization": "wrist_origin_maxabs"}, out)

    # ── Held-out test evaluation ──────────────────────────────────────────────
    pt = predict(model, X_te)
    X_te_mirror = np.stack([mirror(x) for x in X_te])
    pm = predict(model, X_te_mirror)
    present = sorted(set(y_te))
    report = classification_report(y_te, pt, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pt, labels=list(range(len(labels))))

    metrics = {
        "test_accuracy": accuracy_score(y_te, pt),
        "test_macro_f1": f1_score(y_te, pt, labels=present, average="macro", zero_division=0),
        "test_accuracy_mirrored": accuracy_score(y_te, pm),
        "best_val_macro_f1": best[0],
        "best_val_accuracy": best[1],
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "augment": not args.no_augment,
        "aug_strength": args.aug_strength,
        "class_weights": not args.no_class_weights,
        "split": {"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
        "labels": labels,
        "class_counts": {l: counts[l] for l in labels},
        "per_class": {l: {k: report[l][k] for k in ("precision", "recall", "f1-score", "support")}
                      for l in labels},
        "confusion_matrix": cm.tolist(),
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))

    print(f"\nBest epoch {best_epoch}: val macro-F1 {best[0]*100:.1f}%, val acc {best[1]*100:.1f}%")
    print("\nTest set (never seen during training or model selection):")
    print(classification_report(y_te, pt, labels=present,
                                target_names=[labels[i] for i in present], zero_division=0))
    print(f"Test accuracy             {metrics['test_accuracy']*100:.1f}%")
    print(f"Test macro-F1             {metrics['test_macro_f1']*100:.1f}%")
    print(f"Mirrored (left-hand) acc  {metrics['test_accuracy_mirrored']*100:.1f}%")
    print(f"\nSaved model   -> {out}")
    print(f"Saved metrics -> {METRICS_PATH}")

    if not args.no_plots:
        save_plots(history, cm, labels, best_epoch)
        print(f"Saved plots   -> {STATIC_DIR}/training_curves.png, confusion_matrix.png")
    print("\nNext: streamlit run app.py")


if __name__ == "__main__":
    main()
