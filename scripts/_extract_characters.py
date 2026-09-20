"""Unpack downloaded MMD character archives into ``assets/`` and find each model.

Two things make this more than ``unzip``. MMD packages are usually zipped on Japanese
or Chinese Windows, so entry names are raw bytes that ``zipfile`` hands back as CP437
mojibake — decoding has to be retried through CP932 and GBK or the extracted tree is
unreadable and the PMX cannot find its textures. And the archives have no common
layout: some hold the model at the root, some nest it two or three directories deep
beside a readme and a PSD.

Nothing here is committed; ``assets/`` is ignored, and these models carry their own
redistribution terms.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

#: Tried in order. ``gbk`` comes before ``cp932`` deliberately: these packs are
#: Chinese, and a GBK name misread as CP932 still *decodes* — it just produces
#: half-width katakana. ``颜.png`` arriving as ``ﾑﾕ.png`` is exactly that, and because
#: it decodes cleanly a cp932-first order stops there and never tries GBK, so every
#: texture the PMX asks for by its real name goes missing and the model renders
#: magenta. Preferring GBK, and only accepting a decode whose CJK actually lands in
#: the CJK block, gets the real name back.
CODECS = ("gbk", "cp932", "utf-8")

#: Half-width katakana, the signature of a GBK name misread as Shift-JIS.
_KATAKANA = range(0xFF61, 0xFFA0)


def _plausible(decoded: str) -> bool:
    """Reject a decode that is all half-width katakana — that is one failure mode."""
    letters = [ch for ch in decoded if not ch.isascii()]
    if not letters:
        return True
    return not all(ord(ch) in _KATAKANA for ch in letters)


#: Both directions of the same mistake. A Chinese pack writes GBK and a reader assuming
#: Shift-JIS turns 颜 into ﾑﾕ; a Japanese pack writes Shift-JIS and a reader assuming GBK
#: turns 体 into 懱. Neither raises, and neither result looks wrong on its own — 懱 is a
#: perfectly real ideograph — so a name cannot be judged in isolation.
_REPAIRS = (("cp932", "gbk"), ("gbk", "cp932"), ("shift_jis", "gbk"), ("gbk", "shift_jis"))


def repair_against(names: list[str], wanted: set[str]) -> dict[str, str]:
    """Rename files already on disk to the names the PMX asks for.

    The only dependable test is external. Scoring a candidate by how ordinary its
    characters look does not work: ``懱`` and its repair ``体`` are both in the CJK block,
    both are single ideographs, and both look equally plausible. What separates them is
    that the model references ``体.png`` and never ``懱.png``.

    So the repair is driven by the PMX's own texture list: for each file on disk, try both
    directions of the mojibake and keep the one the model is looking for. A file nothing
    references is left alone rather than guessed at.
    """
    mapping = {}
    for name in names:
        if name in wanted:
            continue
        for source, target in _REPAIRS:
            try:
                candidate = name.encode(source).decode(target)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate in wanted and _plausible(candidate):
                mapping[name] = candidate
                break
    return mapping


def decode(name: str) -> str:
    """Recover a mojibake entry name, or return it unchanged if it is already fine.

    ``gbk`` is tried before ``cp932`` deliberately: these packs are mostly Chinese, and a
    GBK name misread as CP932 still *decodes* — it just produces half-width katakana.
    ``颜.png`` arriving as ``ﾑﾕ.png`` is exactly that, and because it decodes cleanly a
    cp932-first order stops there and never tries GBK, so every texture the PMX asks for
    by its real name goes missing and the model renders magenta.
    """
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        # Already decoded by zipfile, but possibly through the wrong codec.
        for source, target in _REPAIRS:
            try:
                candidate = name.encode(source).decode(target)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _plausible(candidate):
                return candidate
        return name
    for codec in CODECS:
        try:
            decoded = raw.decode(codec)
        except UnicodeDecodeError:
            continue
        if decoded.isprintable() and _plausible(decoded):
            return decoded
    return name


def slug(archive: Path) -> str:
    """A short directory name: the part before ``_by_``, punctuation stripped."""
    stem = archive.stem.split("_by_")[0]
    return "".join(ch for ch in stem if ch not in "【】（）()《》 ").strip() or archive.stem[:16]


def extract(archive: Path, into: Path) -> list[Path]:
    if archive.suffix.lower() != ".zip":
        raise ValueError(f"only zip is handled here, got {archive.suffix}")
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            name = decode(info.filename)
            target = into / name
            # Never let an archive escape its own directory.
            if not str(target.resolve()).startswith(str(into.resolve())):
                print(f"    skipped suspicious path {name!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                sink.write(source.read())
    return sorted(into.rglob("*.pmx"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--into", type=Path, default=Path("assets/fbx/pmx"))
    args = parser.parse_args()

    archives = sorted(
        path
        for path in args.downloads.iterdir()
        if path.suffix.lower() in (".zip", ".rar") and not path.name.startswith(("SMPL", "AAAI", "sonic"))
    )
    print(f"{len(archives)} archive(s)\n")
    unhandled: list[Path] = []
    for archive in archives:
        name = slug(archive)
        if archive.suffix.lower() != ".zip":
            unhandled.append(archive)
            print(f"{name}: SKIP {archive.suffix} (no extractor available)")
            continue
        into = args.into / name
        if into.exists() and any(into.rglob("*.pmx")):
            found = sorted(into.rglob("*.pmx"))
            print(f"{name}: already there, {len(found)} pmx")
            continue
        try:
            found = extract(archive, into)
        except Exception as exc:  # noqa: BLE001 - one bad archive must not stop the rest
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"{name}: {len(found)} pmx")
        for pmx in found:
            print(f"    {pmx.relative_to(args.into)}")
    if unhandled:
        print(f"\n{len(unhandled)} archive(s) need a different extractor:")
        for path in unhandled:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
