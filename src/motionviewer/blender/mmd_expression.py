"""Facial expression presets for imported MMD characters.

MMD ships expressions as vertex morphs, which arrive as Blender shape keys. The
importer leaves every one at zero, so an untouched face is the model's *neutral*
mesh — and a neutral anime face with a visible mouth line reads as a grimace next
to the same character smiling. Nothing was switched on by accident; nothing was
switched on at all.

Morph names are a de facto standard across the MMD ecosystem, and the models
tested carry the same vocabulary: ``にこり`` narrows the eyes into a gentle smile,
``口角上げ`` lifts the mouth corners, ``笑い`` closes the eyes into happy arcs.
Each slot below lists aliases, so a model that names one of them differently still
resolves, and a slot that resolves to nothing is reported rather than ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Each entry is (aliases, weight). First alias present on the model wins.
_SMILE_EYES = (("にこり", "ニコリ", "笑い", "smile"), 0.62)
_SMILE_MOUTH = (("口角上げ", "にっこり", "スマイル", "smile2"), 0.55)
_JOY_EYES = (("笑い", "にこり"), 1.0)
_BROW_SOFT = (("にこり左", "真面目"), 0.35)

EXPRESSION_PRESETS: dict[str, tuple[tuple[tuple[str, ...], float], ...]] = {
    # A closed-mouth smile. The default because it survives any camera angle and
    # any motion, where an open-mouth expression starts to look like shouting.
    "smile": (_SMILE_EYES, _SMILE_MOUTH),
    # Half strength, for shots where the face is small and a full smile reads as
    # a squint.
    "soft_smile": ((_SMILE_EYES[0], 0.36), (_SMILE_MOUTH[0], 0.32)),
    # Eyes closed into arcs. Cheerful, but it hides the eyes.
    "happy": (_JOY_EYES, (_SMILE_MOUTH[0], 0.5)),
    "calm": ((_SMILE_MOUTH[0], 0.22), _BROW_SOFT),
    "neutral": (),
}


@dataclass(frozen=True)
class ExpressionResult:
    applied: dict[str, float]
    missing: list[str]
    shape_keys_available: int


def _resolve(block_names: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in block_names:
            return alias
    return None


def apply_expression(meshes: list[Any], preset: str, *, amount: float = 1.0) -> dict:
    """Set the shape keys for ``preset`` on every mesh that has them.

    ``amount`` scales the whole preset, so one knob dials a smile up or down
    without re-tuning each morph.
    """
    if preset not in EXPRESSION_PRESETS:
        raise ValueError(f"unknown expression {preset!r}; known: {sorted(EXPRESSION_PRESETS)}")
    slots = EXPRESSION_PRESETS[preset]

    applied: dict[str, float] = {}
    missing: list[str] = []
    available = 0
    for mesh in meshes:
        keys = getattr(mesh.data, "shape_keys", None)
        if keys is None:
            continue
        blocks = {block.name: block for block in keys.key_blocks}
        available = max(available, len(blocks))
        # Start from neutral so repeated calls do not stack.
        for block in blocks.values():
            if block.name != "Basis":
                block.value = 0.0
        for aliases, weight in slots:
            name = _resolve(set(blocks), aliases)
            if name is None:
                missing.append(aliases[0])
                continue
            value = max(0.0, min(1.0, weight * float(amount)))
            blocks[name].value = value
            applied[name] = round(value, 3)

    return {
        "preset": preset,
        "amount": float(amount),
        "applied": applied,
        "missing": sorted(set(missing)),
        "shape_keys_available": available,
    }
