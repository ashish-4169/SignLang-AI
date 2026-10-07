"""
Landmark augmentation for training.

Works on normalized 63-dim vectors and re-normalizes afterwards, so augmented
samples look exactly like what the app feeds the model at inference time.
"""

import numpy as np

from .landmarks import normalize_landmarks


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def mirror(features: np.ndarray) -> np.ndarray:
    """Flip left/right: turns a right-hand sample into a left-hand one."""
    c = np.asarray(features, dtype=np.float32).reshape(21, 3).copy()
    c[:, 0] *= -1
    return c.flatten()


def augment(features: np.ndarray, rng: np.random.Generator,
            p_mirror: float = 0.5, max_roll_deg: float = 15.0, max_tilt_deg: float = 10.0,
            stretch: float = 0.1, jitter: float = 0.015) -> np.ndarray:
    """One random augmentation of a single sample.

    - mirror: left/right hand invariance
    - roll (in-plane) + small tilt rotations: wrist angle / camera angle
    - x/y stretch: perspective and hand-shape differences
    - gaussian jitter: landmark detection noise
    """
    c = np.asarray(features, dtype=np.float32).reshape(21, 3).copy()
    if rng.random() < p_mirror:
        c[:, 0] *= -1
    rz = np.deg2rad(rng.uniform(-max_roll_deg, max_roll_deg))
    rx, ry = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg, size=2))
    c = c @ _rotation(rx, ry, rz).T
    c[:, :2] *= rng.uniform(1 - stretch, 1 + stretch, size=2).astype(np.float32)
    c += rng.normal(0, jitter, size=c.shape).astype(np.float32)
    return normalize_landmarks(c)
