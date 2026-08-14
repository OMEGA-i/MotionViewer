"""Extract an MMD character zip with the right filename encoding, then verify it.

Zip entries from the MMD ecosystem carry no encoding unless the UTF-8 flag is
set, and the two conventions in circulation disagree: a Japanese tool writes
Shift-JIS, a Chinese one writes GBK, and one archive can mix flagged UTF-8 with
unflagged Shift-JIS.  Guessing wrong is quiet and expensive — the PMX still
imports, its textures just resolve to nothing, and the character renders as grey
clay.

So the guess is checked rather than trusted: the PMX's own texture table is
stored in UTF-16 or UTF-8 and is therefore always correct, so it is read back and
compared against what landed on disk.

  uv run python scripts/_extract_pmx_zip.py <archive.zip> assets/fbx/pmx/<name>
"""

from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

_CANDIDATE_ENCODINGS = ("cp932", "gbk")


def _decode_entry(info: zipfile.ZipInfo) -> list[str]:
    """Candidate names for one entry, best first."""
    if info.flag_bits & 0x800:
        return [info.filename]
    raw = info.filename.encode("cp437", errors="replace")
    names: list[str] = []
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            decoded = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "�" not in decoded and decoded not in names:
            names.append(decoded)
    return names or [info.filename]


def _read_text(buffer: bytes, offset: int, encoding: str) -> tuple[str, int]:
    (size,) = struct.unpack_from("<i", buffer, offset)
    offset += 4
    raw = buffer[offset : offset + size]
    return raw.decode(encoding, errors="replace"), offset + size


def pmx_texture_paths(path: Path) -> list[str]:
    """Texture paths declared inside a PMX, in file order.

    Everything before the texture table is variable width, so vertices and faces
    have to be walked rather than skipped.
    """
    data = path.read_bytes()
    if data[:4] != b"PMX ":
        raise ValueError(f"not a PMX file: {path}")
    globals_size = data[8]
    globals_bytes = data[9 : 9 + globals_size]
    encoding = "utf-16-le" if globals_bytes[0] == 0 else "utf-8"
    additional_uv = globals_bytes[1]
    vertex_index_size = globals_bytes[2]
    offset = 9 + globals_size
    for _ in range(4):  # model name and comment, JP and EN
        _, offset = _read_text(data, offset, encoding)

    (vertex_count,) = struct.unpack_from("<i", data, offset)
    offset += 4
    # position + normal + uv, then optional extra UV layers
    stride = (3 + 3 + 2) * 4 + additional_uv * 16
    weight_extra = {0: 0, 1: 4, 2: 16, 3: 16 + 36, 4: 16}
    bone_index_size = globals_bytes[5]
    weight_bones = {0: 1, 1: 2, 2: 4, 3: 2, 4: 4}
    for _ in range(vertex_count):
        offset += stride
        weight_type = data[offset]
        offset += 1
        offset += bone_index_size * weight_bones[weight_type]
        offset += weight_extra[weight_type]
        offset += 4  # edge scale

    (index_count,) = struct.unpack_from("<i", data, offset)
    offset += 4 + index_count * vertex_index_size

    (texture_count,) = struct.unpack_from("<i", data, offset)
    offset += 4
    textures: list[str] = []
    for _ in range(texture_count):
        name, offset = _read_text(data, offset, encoding)
        textures.append(name.replace("\\", "/"))
    return textures


def extract(archive: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    with zipfile.ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            candidates = _decode_entry(info)
            chosen = candidates[0]
            target = destination / chosen
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(info) as source, open(target, "wb") as out:
                out.write(source.read())
            written[chosen] = candidates
    return written


def verify(destination: Path, written: dict[str, list[str]]) -> dict:
    """Check every PMX's texture table against the files on disk, and repair."""
    report: dict = {"models": [], "renamed": [], "missing": []}
    on_disk = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    alternates: dict[str, list[str]] = {name: names[1:] for name, names in written.items() if len(names) > 1}

    for pmx in sorted(destination.rglob("*.pmx")):
        base = pmx.parent
        try:
            textures = pmx_texture_paths(pmx)
        except Exception as exc:  # A broken PMX must be reported, not swallowed.
            report["models"].append({"pmx": pmx.relative_to(destination).as_posix(), "error": str(exc)})
            continue
        missing: list[str] = []
        for texture in textures:
            wanted = (base / texture).resolve()
            if wanted.is_file():
                continue
            # The entry may have been written under a different candidate name.
            relative = (base.relative_to(destination) / texture).as_posix()
            repaired = False
            for name, others in alternates.items():
                if relative in others or Path(relative).name in {Path(o).name for o in others}:
                    source = destination / name
                    if source.is_file():
                        wanted.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(wanted)
                        report["renamed"].append({"from": name, "to": relative})
                        repaired = True
                        break
            if not repaired:
                missing.append(texture)
        report["models"].append(
            {
                "pmx": pmx.relative_to(destination).as_posix(),
                "textures": len(textures),
                "missing": len(missing),
            }
        )
        report["missing"].extend(f"{pmx.name}: {name}" for name in missing)
    report["files_on_disk"] = len(on_disk)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    written = extract(args.archive, args.destination)
    report = verify(args.destination, written)
    print(f"extracted {len(written)} files -> {args.destination}")
    for entry in report["renamed"]:
        print(f"  repaired name: {entry['from']} -> {entry['to']}")
    for model in report["models"]:
        if "error" in model:
            print(f"  {model['pmx']}: PARSE ERROR {model['error']}")
        else:
            state = "ok" if model["missing"] == 0 else f"MISSING {model['missing']}"
            print(f"  {model['pmx']}: {model['textures']} textures, {state}")
    for name in report["missing"][:20]:
        print(f"    missing: {name}")


if __name__ == "__main__":
    main()
