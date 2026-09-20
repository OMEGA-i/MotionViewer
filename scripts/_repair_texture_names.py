"""Rename texture files that were extracted through the wrong codec.

Three character directories predate the decoder fix and were skipped by the
re-extraction because they already existed. Furina renders magenta as a result: the PMX
asks for ``体.png`` and ``髮.png``, and the files on disk are named ``懱.png`` and
``闆.png`` — the same bytes read as GBK instead of Shift-JIS.

The repair is driven by the PMX's own texture list rather than by judging the names.
Scoring them does not work: ``懱`` and ``体`` are both ordinary CJK ideographs, both
single characters, both equally plausible in isolation. What separates them is that the
model references one and never the other.

  uv run python scripts/_repair_texture_names.py --dry-run
  uv run python scripts/_repair_texture_names.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _extract_characters import repair_against  # noqa: E402

#: Texture extensions a PMX can reference.
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds", ".spa", ".sph"}


def texture_names(pmx: Path) -> set[str]:
    """Filenames the PMX references, read out of its raw bytes.

    The texture table is UTF-16LE in a PMX 2.0 file, but paths also appear inside
    material blocks, so rather than parse the format this scans for anything that looks
    like a filename with a known extension. A few false positives cost nothing — the
    repair only fires when a candidate rename matches one of them exactly.
    """
    blob = pmx.read_bytes()
    found: set[str] = set()
    for encoding in ("utf-16-le", "utf-8", "cp932", "gbk"):
        try:
            text = blob.decode(encoding, errors="ignore")
        except LookupError:
            continue
        pattern = r"[^\x00-\x1f\\/:*?\"<>|]{1,64}\.(?:" + "|".join(s[1:] for s in SUFFIXES) + ")"
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            found.add(Path(match).name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "assets/fbx/pmx")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        wanted: set[str] = set()
        for pmx in directory.rglob("*.pmx"):
            wanted |= texture_names(pmx)
        if not wanted:
            continue
        files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES]
        mapping = repair_against([p.name for p in files], wanted)
        if not mapping:
            continue
        print(f"{directory.name}: {len(mapping)} file(s) to rename", flush=True)
        for path in files:
            new_name = mapping.get(path.name)
            if not new_name:
                continue
            target = path.with_name(new_name)
            print(f"    {path.name}  ->  {new_name}")
            total += 1
            if args.dry_run:
                continue
            if target.exists():
                print("      target already exists, skipped")
                continue
            path.rename(target)
    verb = "would rename" if args.dry_run else "renamed"
    print(f"\n{verb} {total} file(s)")


if __name__ == "__main__":
    main()
