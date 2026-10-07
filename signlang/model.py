"""GestureNet: a small MLP over normalized hand landmarks, shared by training and the app."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 63  # 21 landmarks x (x, y, z)


class GestureNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GestureClassifier:
    """Loads a trained checkpoint and predicts (label, confidence) for one hand."""

    def __init__(self, model: GestureNet, labels: list[str]):
        self.model = model.eval()
        self.labels = labels

    @classmethod
    def load(cls, path: str | Path) -> "GestureClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        labels = list(ckpt["labels"])
        model = GestureNet(ckpt.get("input_dim", INPUT_DIM), len(labels))
        model.load_state_dict(ckpt["model_state"])
        return cls(model, labels)

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(-1, INPUT_DIM)
        return torch.softmax(self.model(x), dim=1).numpy()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(features)[0]
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])
