"""Does the cloth move *with* the body, or on its own?

"Unnatural shaking that does not match the character's movement" is a complaint about
correlation, not about amplitude, so amplitude alone cannot answer it. Two failures look
similar in a still and are opposites in motion:

- **Lag that is too slow or too fast.** Cloth should trail the body by a few frames. If
  the peak correlation sits at zero lag the cloth is welded to the skeleton; if it sits
  far out, or the correlation is weak at every lag, the cloth is ringing on its own.
- **High-frequency chatter.** A solver that is unstable adds motion above any frequency a
  garment can physically have. Measured as the share of frame-to-frame acceleration in
  the top half of the band.

This compares the two renders of the same clip, so a difference is attributable to the
physics rather than to the motion.

  uv run python scripts/_probe_cloth_motion.py --off /tmp/x_off --on /tmp/x_on
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def silhouette_columns(path: Path) -> np.ndarray:
    """Per-row horizontal extent of the figure: a cheap shape signature per frame."""
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    body = saturation > 22
    return body.sum(axis=1).astype(np.float64)


def series(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """(body centre of mass per frame, lower-body width per frame)."""
    frames = sorted(directory.rglob("frame_*.png"))
    centres, widths = [], []
    for frame in frames:
        rows = silhouette_columns(frame)
        total = rows.sum()
        if total <= 0:
            centres.append(0.0)
            widths.append(0.0)
            continue
        index = np.arange(len(rows))
        centres.append(float((rows * index).sum() / total))
        # Lower half of the figure: where a skirt or a coat lives.
        occupied = np.nonzero(rows > 0)[0]
        if len(occupied) == 0:
            widths.append(0.0)
            continue
        top, bottom = occupied.min(), occupied.max()
        band = rows[top + (bottom - top) // 2 : bottom]
        widths.append(float(band.mean()) if len(band) else 0.0)
    return np.array(centres), np.array(widths)


def high_frequency_share(signal: np.ndarray) -> float:
    """Fraction of spectral energy above half the Nyquist frequency."""
    if len(signal) < 8:
        return 0.0
    detrended = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    if spectrum.sum() <= 0:
        return 0.0
    return float(spectrum[len(spectrum) // 2 :].sum() / spectrum.sum())


def best_lag(body: np.ndarray, cloth: np.ndarray, limit: int = 12) -> tuple[int, float]:
    """Lag in frames at which cloth best follows the body, and that correlation."""
    body = body - body.mean()
    cloth = cloth - cloth.mean()
    if body.std() < 1e-9 or cloth.std() < 1e-9:
        return 0, 0.0
    best, best_value = 0, -2.0
    for lag in range(0, limit + 1):
        a = body[: len(body) - lag]
        b = cloth[lag:]
        if len(a) < 8:
            break
        value = float(np.corrcoef(a, b)[0, 1])
        if value > best_value:
            best, best_value = lag, value
    return best, best_value


def report(label: str, directory: Path) -> None:
    body, cloth = series(directory)
    lag, correlation = best_lag(body, cloth)
    print(
        f"{label:10s} frames {len(body):3d}  cloth-follows-body lag {lag:2d} "
        f"r={correlation:+.2f}  cloth HF share {high_frequency_share(cloth) * 100:5.1f}%  "
        f"body HF share {high_frequency_share(body) * 100:5.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    args = parser.parse_args()
    print("lag 0 = cloth welded to the skeleton; healthy cloth trails a few frames.")
    print("HF share above roughly 25% is chatter no garment can physically have.\n")
    report("physics off", args.off)
    report("physics on", args.on)


if __name__ == "__main__":
    main()
