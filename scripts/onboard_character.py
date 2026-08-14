"""Take a character from archive to verified, in one command. Local helper.

Every new model has to clear the same gates before it is worth rendering, and each
gate has caught a real failure at least once:

1. **textures** — MMD archives carry no filename encoding, and guessing wrong is
   silent: the PMX imports and renders as grey clay. Checked against the PMX's own
   texture table.
2. **mapping** — a Honkai rig has a two-segment upper body where a Genshin rig has
   three, and unknown bone names must fail loudly rather than animate half a body.
3. **rest alignment** — the arm-vs-torso rest gap is what decides each bone's
   transfer mode. A rig whose gaps look unlike the others needs a look before it
   is trusted.
4. **numbers** — the retarget must agree with closed-form theory and with
   Blender's own evaluation.
5. **pictures** — an identity T-pose proves the calibration, and a posed frame
   proves the shading.

  uv run python scripts/onboard_character.py --archive ~/Downloads/model.zip --name furina
  uv run python scripts/onboard_character.py --asset assets/fbx/pmx/furina/x.pmx --name furina
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / ".local/Blender.app/Contents/MacOS/Blender"
DEFAULT_MOTION = ROOT / "data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz"

# Arms disagree with SMPL-X rest by this much on every Genshin/Honkai rig seen so
# far; a rig outside the band is not necessarily broken, but it is not routine.
_EXPECTED_ARM_GAP_DEGREES = (18.0, 60.0)
_EXPECTED_TORSO_GAP_DEGREES = 12.0


def _run(command: list[str], label: str) -> tuple[bool, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
        print(f"  {label}: FAILED\n{tail}")
        return False, result.stdout
    return True, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None, help="Zip to extract first")
    parser.add_argument("--asset", type=Path, default=None, help="PMX to check (skips extraction)")
    parser.add_argument("--name", required=True, help="Short slug, used for asset and output paths")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/onboard")
    parser.add_argument("--frames", type=int, default=40, help="0 checks the whole clip")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    if args.asset is not None:
        args.asset = args.asset.resolve()

    destination = ROOT / "assets/fbx/pmx" / args.name
    report: dict = {"name": args.name, "gates": {}}
    out = args.output / args.name
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. textures ------------------------------------------------------
    if args.archive is not None:
        print(f"[1/5] extracting {args.archive.name}")
        ok, stdout = _run(
            [
                sys.executable,
                str(ROOT / "scripts/_extract_pmx_zip.py"),
                str(args.archive),
                str(destination),
            ],
            "extract",
        )
        report["gates"]["textures"] = "ok" if ok and "MISSING" not in stdout else "failed"
        print(stdout.strip())
        if not ok:
            raise SystemExit(1)

    asset = args.asset
    if asset is None:
        candidates = sorted(destination.rglob("*.pmx"), key=lambda path: -path.stat().st_size)
        if not candidates:
            raise SystemExit(f"no .pmx under {destination}")
        asset = candidates[0]
        print(f"      largest PMX chosen: {asset.resolve().relative_to(ROOT)}")
    report["asset"] = str(asset.resolve().relative_to(ROOT))

    # ---- 2 + 3. mapping and rest alignment -------------------------------
    print("[2/5] rig inspection")
    dump_path = out / "rig_dump.json"
    ok, _ = _run(
        [
            str(BLENDER),
            "--background",
            "--python",
            str(ROOT / "scripts/_dump_mmd_rig.py"),
            "--",
            "--asset",
            str(asset),
            "--motion",
            str(args.motion),
            "--output",
            str(dump_path),
        ],
        "rig dump",
    )
    if not ok:
        raise SystemExit(1)
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    report["bones"] = dump["bone_count"]
    report["errors"] = dump["errors"]
    report["gates"]["mapping"] = "ok" if not dump["errors"] else "failed"
    print(f"      {dump['bone_count']} bones, mapping errors: {dump['errors'] or 'none'}")
    if dump["errors"]:
        raise SystemExit(1)

    alignment = dump["rest_alignment"]
    arms = {
        name: entry["rest_angle_deg"]
        for name, entry in alignment.items()
        if any(token in name for token in ("shoulder", "elbow", "wrist"))
    }
    torso = {
        name: entry["rest_angle_deg"]
        for name, entry in alignment.items()
        if any(token in name for token in ("spine", "neck", "hip", "knee"))
    }
    arm_ok = all(
        _EXPECTED_ARM_GAP_DEGREES[0] <= value <= _EXPECTED_ARM_GAP_DEGREES[1] for value in arms.values()
    )
    torso_ok = all(value <= _EXPECTED_TORSO_GAP_DEGREES for value in torso.values())
    report["rest_gap_arms_deg"] = {name: round(value, 1) for name, value in sorted(arms.items())}
    report["rest_gap_torso_deg"] = {name: round(value, 1) for name, value in sorted(torso.items())}
    report["gates"]["rest_alignment"] = "ok" if arm_ok and torso_ok else "unusual"
    print(
        f"      arm rest gap {min(arms.values()):.0f}-{max(arms.values()):.0f} deg"
        f" ({'routine' if arm_ok else 'UNUSUAL'}),"
        f" torso max {max(torso.values()):.0f} deg ({'routine' if torso_ok else 'UNUSUAL'})"
    )
    print(f"      twist pairs: {[pair['twist'] for pair in dump['twist_pairs']]}")

    # ---- 4. numbers ------------------------------------------------------
    print("[3/5] numeric validation")
    validation_path = out / "validation.json"
    ok, stdout = _run(
        [
            str(BLENDER),
            "--background",
            "--python",
            str(ROOT / "scripts/_validate_mmd_retarget.py"),
            "--",
            "--asset",
            str(asset),
            "--motion",
            str(args.motion),
            "--frames",
            str(args.frames),
            "--output",
            str(validation_path),
        ],
        "validation",
    )
    if not ok:
        raise SystemExit(1)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    worst = {key: round(float(value["value"]), 4) for key, value in validation["worst"].items()}
    report["worst"] = worst
    # Blender evaluates poses in single precision, so ~0.1 deg is the floor.
    gate = worst.get("solver_vs_theory_deg", 1.0) < 1e-3 and worst.get("blender_vs_solver_deg", 1.0) < 0.2
    report["gates"]["numbers"] = "ok" if gate else "failed"
    print(f"      {json.dumps(worst)}")

    # ---- 5. pictures -----------------------------------------------------
    if not args.skip_render:
        print("[4/5] identity T-pose")
        _run(
            [
                str(BLENDER),
                "--background",
                "--python",
                str(ROOT / "scripts/_render_mmd_sheet.py"),
                "--",
                "--asset",
                str(asset),
                "--motion",
                str(args.motion),
                "--output",
                str(out / "identity"),
                "--views",
                "front",
                "--frames",
                "1",
                "--identity",
                "--resolution",
                "700",
                "--toon",
            ],
            "identity render",
        )
        print("[5/5] posed frame")
        _run(
            [
                str(BLENDER),
                "--background",
                "--python",
                str(ROOT / "scripts/_render_mmd_sheet.py"),
                "--",
                "--asset",
                str(asset),
                "--motion",
                str(args.motion),
                "--output",
                str(out / "posed"),
                "--views",
                "front,three_quarter",
                "--frames",
                "14",
                "--in-place",
                "--resolution",
                "700",
                "--toon",
                "--outline",
                "--ground",
            ],
            "posed render",
        )

    (out / "onboard.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "READY" if all(state == "ok" for state in report["gates"].values()) else "REVIEW"
    print(f"\n{args.name}: {verdict}  {json.dumps(report['gates'], ensure_ascii=False)}")
    print(f"  {out / 'onboard.json'}")


if __name__ == "__main__":
    main()
