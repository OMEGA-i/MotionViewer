"""Rank SMPL-X clips by how well they will survive retargeting. Local helper.

Retargeting is exact, so whatever the source does, the character does. That makes
source quality the ceiling on how a result looks, and the two faults seen so far
are not visible in a still frame:

- **angular velocity** past what a body can do. One clip turns at 80 deg per
  frame at 30 fps, 2415 deg/s, which reads as a violent snap on a character with
  hair and a skirt.
- **jitter**: high-frequency wobble in a joint's rotation. The spine is usually
  the worst offender and it carries a rigid head, which is what "the head keeps
  shaking" turns out to be.

Also reported, because they break a shot in their own way: foot skating (a planted
foot sliding) and ground penetration.

  uv run python scripts/score_motion_quality.py --clips .local/soma_all/.../clips/t2m \
      --output outputs/motion_scores.json --top 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motionviewer.core.smplx_fk import (  # noqa: E402
    SMPLX_BODY22_NAMES,
    rodrigues,
    source_to_blender,
)

# Joints whose wobble is most visible on a character: the spine carries the head,
# and the collars carry the arms.
_JITTER_JOINTS = ("spine1", "spine2", "spine3", "neck", "left_collar", "right_collar")
_ANKLES = ("left_ankle", "right_ankle")
_FEET = ("left_foot", "right_foot")


def _rotation_steps(axis_angle: np.ndarray) -> np.ndarray:
    """Per-frame rotation magnitude of one joint, in degrees."""
    matrices = rodrigues(axis_angle)
    steps = np.empty(max(len(matrices) - 1, 0), dtype=np.float64)
    for index in range(1, len(matrices)):
        relative = matrices[index - 1].T @ matrices[index]
        steps[index - 1] = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    return steps


def score_clip(payload: dict, fps: float = 30.0) -> dict:
    """Motion-quality metrics for one clip. Lower ``penalty`` is better."""
    global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
    body_pose = np.asarray(payload["body_pose"], dtype=np.float64).reshape(len(global_orient), 21, 3)
    joints = source_to_blender(np.asarray(payload["joints22"], dtype=np.float64))
    poses = np.concatenate([global_orient[:, None, :], body_pose], axis=1)
    total = len(global_orient)

    index_of = {name: index for index, name in enumerate(SMPLX_BODY22_NAMES)}
    worst_velocity = 0.0
    worst_velocity_joint = ""
    jitter = 0.0
    jitter_joint = ""
    for name, column in index_of.items():
        steps = _rotation_steps(poses[:, column])
        if len(steps) == 0:
            continue
        velocity = float(steps.max()) * fps
        if velocity > worst_velocity:
            worst_velocity, worst_velocity_joint = velocity, name
        if name in _JITTER_JOINTS and len(steps) > 2:
            # Second difference isolates shake from smooth acceleration.
            value = float(np.abs(np.diff(steps)).mean())
            if value > jitter:
                jitter, jitter_joint = value, name

    # Foot skating: horizontal travel of whichever foot is planted, measured only
    # while it is the lower of the two and near the floor.
    floor = float(joints[:, :, 2].min())
    skate = 0.0
    for ankle, toe in zip(_ANKLES, _FEET, strict=True):
        points = joints[:, index_of[toe]]
        height = points[:, 2] - floor
        planted = height < (np.percentile(height, 25) + 0.02)
        travel = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        mask = planted[:-1] & planted[1:]
        if mask.any():
            skate = max(skate, float(travel[mask].max()) * fps)
        _ = ankle

    penetration = max(0.0, floor - float(joints[0, :, 2].min()))
    height_span = float(joints[:, :, 2].max() - floor)

    # Expressiveness. Penalty alone ranks a clip of someone standing still first,
    # because the cleanest motion is no motion; a showcase needs the opposite.
    # Measured root-relative so walking across the room does not count as
    # gesturing.
    root = joints[:, index_of["pelvis"]][:, None, :]
    relative = joints - root
    limbs = [index_of[name] for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "head")]
    reach = float(np.linalg.norm(relative[:, limbs] - relative[:, limbs].mean(axis=0), axis=-1).mean())
    speed = float(np.linalg.norm(np.diff(relative[:, limbs], axis=0), axis=-1).mean()) * fps
    travel = float(np.linalg.norm(joints[-1, index_of["pelvis"], :2] - joints[0, index_of["pelvis"], :2]))

    # Weights chosen so that each term is ~1.0 at the point it becomes visible:
    # 600 deg/s is a fast but legible turn, 0.5 deg of jitter shows on a head,
    # 0.35 m/s of skate reads as sliding.
    penalty = (
        max(0.0, worst_velocity - 600.0) / 600.0
        + jitter / 0.5
        + max(0.0, skate - 0.35) / 0.35
        + penetration / 0.05
    )
    return {
        "frames": total,
        "limb_reach_m": reach,
        "limb_speed_m_s": speed,
        "root_travel_m": travel,
        "activity": float(reach * 2.0 + speed * 0.5),
        "max_angular_velocity_deg_s": worst_velocity,
        "max_angular_velocity_joint": worst_velocity_joint,
        "jitter_deg": jitter,
        "jitter_joint": jitter_joint,
        "foot_skate_m_s": skate,
        "ground_penetration_m": penetration,
        "height_span_m": height_span,
        "penalty": float(penalty),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--source", default="gt")
    parser.add_argument("--meta", type=Path, default=None, help="Directory holding clip meta.json files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-frames", type=int, default=45)
    parser.add_argument("--max-penalty", type=float, default=0.25, help="Quality bar for the shortlist")
    args = parser.parse_args()

    captions: dict[str, str] = {}
    if args.meta is not None:
        for meta_path in args.meta.rglob("meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            captions[str(meta.get("rec_id", ""))] = str(meta.get("caption") or "")

    results: list[dict] = []
    clips = sorted(path for path in args.clips.glob("test_rec_*") if path.is_dir())
    for index, clip in enumerate(clips, start=1):
        motion = clip / args.source / "smplx_params.npz"
        if not motion.is_file():
            continue
        with np.load(motion, allow_pickle=False) as payload:
            data = {key: payload[key] for key in ("global_orient", "body_pose", "joints22")}
        score = score_clip(data)
        if score["frames"] < args.min_frames:
            continue
        rec = clip.name.replace("test_", "")
        score["clip"] = clip.name
        score["rec_id"] = rec
        score["caption"] = captions.get(rec, "")
        results.append(score)
        if index % 100 == 0:
            print(f"  scored {index}/{len(clips)}")

    # Clean first, then the most expressive among the clean. A clip that fails
    # the quality bar is not rescued by being interesting.
    clean = [item for item in results if item["penalty"] <= args.max_penalty]
    clean.sort(key=lambda item: -item["activity"])
    rejected = sorted(
        (item for item in results if item["penalty"] > args.max_penalty),
        key=lambda item: item["penalty"],
    )
    results = clean + rejected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"count": len(results), "clean": len(clean), "max_penalty": args.max_penalty, "clips": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nscored {len(results)} clips, {len(clean)} under the quality bar -> {args.output}")
    print(f"\n{'rank':>4} {'activity':>8} {'penalty':>8} {'deg/s':>7} {'jitter':>7} {'frames':>6}  caption")
    for rank, item in enumerate(results[: args.top], start=1):
        print(
            f"{rank:>4} {item['activity']:8.2f} {item['penalty']:8.2f} "
            f"{item['max_angular_velocity_deg_s']:7.0f} {item['jitter_deg']:7.2f} {item['frames']:6d}  "
            f"{item['caption'][:74]}"
        )


if __name__ == "__main__":
    main()
