from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .errors import PackageFormatError, PackagePayloadError


@dataclass(frozen=True)
class PackageMemberInfo:
    relpath: PurePosixPath
    size: int
    is_file: bool


class PackageStore(Protocol):
    path: Path

    def exists(self, relpath: PurePosixPath | str) -> bool: ...

    def list_files(self, prefix: PurePosixPath | str = "") -> list[PackageMemberInfo]: ...

    def read_bytes(self, relpath: PurePosixPath | str) -> bytes: ...

    def read_json(self, relpath: PurePosixPath | str) -> dict: ...

    def materialize(self, assets: Iterable[tuple[PurePosixPath, Path]]) -> dict[PurePosixPath, Path]: ...

    def close(self) -> None: ...


def open_package_store(path: str | Path) -> PackageStore:
    p = Path(path)
    if not p.exists():
        raise PackageFormatError(f"Package path does not exist: {p}")
    if p.is_dir():
        return DirectoryPackageStore(p)
    if _is_tar_package(p):
        return TarPackageStore(p)
    raise PackageFormatError(f"Not a package directory or tar.gz: {p}")


def probe_package(path: str | Path) -> bool:
    p = Path(path)
    try:
        store = open_package_store(p)
    except PackageFormatError:
        return False
    try:
        if not store.exists("manifest.json"):
            return False
        manifest = store.read_json("manifest.json")
        return str(manifest.get("protocol_version", "")) == "2.0"
    except Exception:
        return False
    finally:
        store.close()


class DirectoryPackageStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self._root = self._detect_root(self.path)

    @property
    def root(self) -> Path:
        return self._root

    def exists(self, relpath: PurePosixPath | str) -> bool:
        target = self._resolve(relpath)
        return target.is_file()

    def list_files(self, prefix: PurePosixPath | str = "") -> list[PackageMemberInfo]:
        prefix_path = PurePosixPath(str(prefix)) if prefix else PurePosixPath(".")
        base = self._root if str(prefix_path) in ("", ".") else self._resolve(prefix_path, expect_file=False)
        if not base.exists():
            return []
        members: list[PackageMemberInfo] = []
        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            rel = PurePosixPath(file_path.relative_to(self._root).as_posix())
            members.append(PackageMemberInfo(relpath=rel, size=file_path.stat().st_size, is_file=True))
        return members

    def read_bytes(self, relpath: PurePosixPath | str) -> bytes:
        return self._resolve(relpath).read_bytes()

    def read_json(self, relpath: PurePosixPath | str) -> dict:
        try:
            return json.loads(self.read_bytes(relpath).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageFormatError(f"Invalid JSON at {relpath}: {exc}") from exc

    def materialize(self, assets: Iterable[tuple[PurePosixPath, Path]]) -> dict[PurePosixPath, Path]:
        results: dict[PurePosixPath, Path] = {}
        for relpath, dest in assets:
            source = self._resolve(relpath)
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != source.stat().st_size:
                shutil.copy2(source, dest)
            results[PurePosixPath(str(relpath))] = dest
        return results

    def close(self) -> None:
        return None

    def _resolve(self, relpath: PurePosixPath | str, *, expect_file: bool = True) -> Path:
        pure = _safe_relpath(relpath)
        target = (self._root / Path(*pure.parts)).resolve()
        if not str(target).startswith(str(self._root)):
            raise PackagePayloadError(f"Unsafe package path escapes root: {relpath}")
        if expect_file and target.exists() and not target.is_file():
            raise PackagePayloadError(f"Expected file asset, found directory: {relpath}")
        return target

    @staticmethod
    def _detect_root(path: Path) -> Path:
        if (path / "manifest.json").is_file():
            return path
        children = [child for child in path.iterdir() if child.is_dir()]
        if len(children) == 1 and (children[0] / "manifest.json").is_file():
            return children[0]
        raise PackageFormatError(f"No manifest.json found under package directory: {path}")


class TarPackageStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            self._archive = tarfile.open(self.path, "r:*")
        except (tarfile.TarError, OSError) as exc:
            raise PackageFormatError(f"Unable to open package tar: {path}: {exc}") from exc
        self._members = {name: info for name, info in ((m.name, m) for m in self._archive.getmembers())}
        self._root_prefix = self._detect_root_prefix()

    def exists(self, relpath: PurePosixPath | str) -> bool:
        name = self._member_name(relpath)
        info = self._members.get(name)
        return info is not None and info.isfile()

    def list_files(self, prefix: PurePosixPath | str = "") -> list[PackageMemberInfo]:
        prefix_path = PurePosixPath(str(prefix)) if prefix else PurePosixPath(".")
        prefix_str = "" if str(prefix_path) in ("", ".") else str(prefix_path).rstrip("/") + "/"
        members: list[PackageMemberInfo] = []
        for info in self._archive.getmembers():
            if not info.isfile():
                continue
            rel = self._rel_from_member(info.name)
            if rel is None:
                continue
            if prefix_str and not str(rel).startswith(prefix_str):
                continue
            members.append(PackageMemberInfo(relpath=rel, size=int(info.size), is_file=True))
        return sorted(members, key=lambda item: str(item.relpath))

    def read_bytes(self, relpath: PurePosixPath | str) -> bytes:
        info = self._require_file(relpath)
        handle = self._archive.extractfile(info)
        if handle is None:
            raise PackagePayloadError(f"Failed to read tar member: {relpath}")
        return handle.read()

    def read_json(self, relpath: PurePosixPath | str) -> dict:
        try:
            return json.loads(self.read_bytes(relpath).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageFormatError(f"Invalid JSON at {relpath}: {exc}") from exc

    def materialize(self, assets: Iterable[tuple[PurePosixPath, Path]]) -> dict[PurePosixPath, Path]:
        results: dict[PurePosixPath, Path] = {}
        seen_dests: set[Path] = set()
        for relpath, dest in assets:
            pure = PurePosixPath(str(relpath))
            dest = Path(dest).resolve()
            if dest in seen_dests:
                raise PackagePayloadError(f"Duplicate materialization target: {dest}")
            seen_dests.add(dest)
            info = self._require_file(pure)
            if info.issym() or info.islnk():
                raise PackagePayloadError(f"Refusing to extract link member: {relpath}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == info.size:
                results[pure] = dest
                continue
            handle = self._archive.extractfile(info)
            if handle is None:
                raise PackagePayloadError(f"Failed to extract tar member: {relpath}")
            with dest.open("wb") as out:
                shutil.copyfileobj(handle, out)
            results[pure] = dest
        return results

    def close(self) -> None:
        self._archive.close()

    def _require_file(self, relpath: PurePosixPath | str):
        name = self._member_name(relpath)
        info = self._members.get(name)
        if info is None:
            raise PackagePayloadError(f"Missing package member: {relpath}")
        if not info.isfile() or info.issym() or info.islnk():
            raise PackagePayloadError(f"Refusing non-regular-file package member: {relpath}")
        return info

    def _member_name(self, relpath: PurePosixPath | str) -> str:
        pure = _safe_relpath(relpath)
        if self._root_prefix:
            return f"{self._root_prefix}/{pure.as_posix()}"
        return pure.as_posix()

    def _rel_from_member(self, member_name: str) -> PurePosixPath | None:
        pure = PurePosixPath(member_name)
        if pure.is_absolute() or ".." in pure.parts:
            return None
        parts = pure.parts
        if self._root_prefix:
            root_parts = PurePosixPath(self._root_prefix).parts
            if parts[: len(root_parts)] != root_parts:
                return None
            parts = parts[len(root_parts) :]
        if not parts:
            return None
        return PurePosixPath(*parts)

    def _detect_root_prefix(self) -> str:
        for name in self._members:
            pure = PurePosixPath(name)
            if pure.name == "manifest.json" and pure.parent != PurePosixPath("."):
                # Prefer nested root/<manifest.json>
                return pure.parent.as_posix()
        if "manifest.json" in self._members:
            return ""
        # Also accept root/manifest.json discovered via normalized names
        for name in self._members:
            pure = PurePosixPath(name)
            if pure.name == "manifest.json":
                parent = pure.parent.as_posix()
                return "" if parent == "." else parent
        raise PackageFormatError(f"No manifest.json found in package tar: {self.path}")


def _safe_relpath(relpath: PurePosixPath | str) -> PurePosixPath:
    pure = PurePosixPath(str(relpath))
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix().startswith("/"):
        raise PackagePayloadError(f"Unsafe package relative path: {relpath}")
    if str(pure) in ("", "."):
        raise PackagePayloadError("Empty package relative path")
    return pure


def _is_tar_package(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"))
