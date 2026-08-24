"""Check every claim the SOMA path rests on, against the real probe clips.

The unit tests use synthetic chains, which prove the maths but not that the files match
the maths. This runs on ``soma_retarget_probe`` and prints the numbers, so the decision
to read SOMA directly can be re-checked whenever the exporter changes.

  uv run python scripts/_validate_soma_retarget.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from motionviewer.core.soma_fk import (  # noqa: E402
    forward_kinematics,
    recover_rest_directions,
    rest_joints_from_directions,
    world_rotations,
)

# Rotation applied as a world rotation instead of chained: the plausible wrong reading.
BAD_READING_LIMIT_DEG = 5.0
ORIENTATION_LIMIT_DEG = 0.1
POSITION_LIMIT_MM = 0.05


def bone_direction_error(joints, rotations, parents, directions) -> np.ndarray:
    out = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        delta = joints[:, child] - joints[:, parent]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        observed = delta / np.maximum(length, 1e-12)
        predicted = np.einsum("tij,j->ti", rotations[:, parent], directions[child])
        out.append(np.degrees(np.arccos(np.clip((predicted * observed).sum(axis=1), -1.0, 1.0))))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=Path("/Users/a26044/Motion data/soma_retarget_probe"))
    args = parser.parse_args()

    files = sorted((args.probe / "soma").glob("*.soma.npz"))
    if not files:
        raise SystemExit(f"no SOMA clips under {args.probe / 'soma'}")
    meta = {c["rec_id"]: c for c in json.loads((args.probe / "meta.json").read_text())["clips"]}

    print(
        f"{'clip':12s} {'orient°':>9s} {'chained-vs-stored°':>19s} {'FK mm':>8s} "
        f"{'rigid mm':>9s} {'height m':>9s} {'SMPL-X fit mm':>14s}"
    )
    print("-" * 88)
    all_directions = []
    failures: list[str] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            joints = np.asarray(data["joints77"], dtype=np.float64)
            local = np.asarray(data["rotmat77"], dtype=np.float64)
            parents = np.asarray(data["joint_parents"], dtype=int)
            lengths = np.asarray(data["bone_lengths"], dtype=np.float64)
            rec = str(data["rec_id"])

        world = world_rotations(local, parents)
        directions = recover_rest_directions(joints, world, parents)
        all_directions.append(directions)

        orient = bone_direction_error(joints, world, parents, directions)
        # The same measurement under the wrong reading, to keep the contrast on record.
        naive_dirs = recover_rest_directions(joints, local, parents)
        naive = bone_direction_error(joints, local, parents, naive_dirs)

        rest = rest_joints_from_directions(directions, lengths, parents)
        rebuilt = forward_kinematics(rest, world, parents, joints[:, 0])
        position_mm = np.linalg.norm(rebuilt - joints, axis=-1).max() * 1000.0

        rigid_mm = 0.0
        for child, parent in enumerate(parents):
            if parent >= 0:
                rigid_mm = max(
                    rigid_mm, float(np.linalg.norm(joints[:, child] - joints[:, parent], axis=1).std()) * 1000
                )
        height = float(rest[:, 1].max() - rest[:, 1].min())
        fit = float(meta.get(rec, {}).get("smplx_fit_rmse_mm", float("nan")))
        print(
            f"{rec[4:14]:12s} {orient.max():8.4f}° {np.median(naive):18.2f}° "
            f"{position_mm:7.4f} {rigid_mm:8.5f} {height:8.3f} {fit:13.1f}"
        )
        if orient.max() > ORIENTATION_LIMIT_DEG:
            failures.append(f"{rec}: bone orientation {orient.max():.3f} deg")
        if position_mm > POSITION_LIMIT_MM:
            failures.append(f"{rec}: rigid FK {position_mm:.3f} mm")
        if np.median(naive) < BAD_READING_LIMIT_DEG:
            failures.append(f"{rec}: chained and stored rotations agree — convention may have changed")

    spread = []
    stacked = np.stack(all_directions)
    for joint in range(1, stacked.shape[1]):
        vectors = stacked[:, joint]
        for a in range(len(vectors)):
            for b in range(a + 1, len(vectors)):
                spread.append(np.degrees(np.arccos(np.clip(vectors[a] @ vectors[b], -1.0, 1.0))))
    print(f"\nrest directions across clips: median {np.median(spread):.4f}°  max {np.max(spread):.4f}°")
    print(
        "orient° = bone direction reproduced from chained rotations; "
        "chained-vs-stored° = the same under the wrong reading"
    )
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
