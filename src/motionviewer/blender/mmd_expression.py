"""Facial expression presets for imported MMD characters.

MMD ships expressions as vertex morphs, which arrive as Blender shape keys. The
importer leaves every one at zero, so an untouched face is the model's *neutral*
mesh — nothing is switched on by accident, and nothing is switched on at all.

**Morphs are grouped, and the group decides what a name does.** A PMX orders its
morphs 眉 (brow), 目 (eye), 口 (mouth), and the same word means different things in
different groups. ``にこり`` sits in the *brow* block: it curves the eyebrows and
does not touch the mouth at all — measured on Yoimiya, driving it to 1.0 changes
exactly **zero** pixels in the mouth region. An earlier version of this file used it
as the "smiling eyes" slot, which is why the smile preset moved almost nothing: it
softened the brows, nudged the mouth, and left the eyes neutral.

What each slot actually needs, verified by rendering face close-ups and counting
changed pixels against the neutral mesh:

- **eyes**: ``なごみ`` narrows them into a gentle smile and is the single biggest
  contributor (12304 changed pixels on its own, against 1537 for the whole old
  preset). ``笑い`` closes them into happy arcs — cheerful, but it hides the eyes,
  so it belongs in a separate preset rather than the default.
- **mouth**: ``口角上げ`` at **0.75**. These faces draw the mouth as *two strokes with
  a gap in the middle*, and that gap is in the neutral mesh, so pushing the corners to
  1.0 stretches both strokes until the pair reads as an open mouth. Stacking ``にやり``
  on top doubles it, because on these models ``にやり`` moves the mouth almost
  identically to ``口角上げ``. The ceiling used to be lower still: what made the
  corners look like fangs was the *outline* painting them near-black, and once that
  was reading the model's own edge colour there was room to lift the corners further.
  Counting changed pixels catches none of this — ``口角上げ`` at 1.0 changes 456 px,
  which reads as "too subtle" in a table and as a grin on screen.
- **brows**: ``にこり`` softens them, worth keeping as a supporting slot.

Alias lists are generous because the vocabulary is only a de facto standard and the
digits vary in width: Yoimiya has ``にやり３`` (full width), Silver Wolf has
``にやり3`` (half width) and no plain ``にやり`` at all. A slot that resolves to
nothing is reported in ``missing`` rather than ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Aliases per slot, most preferred first. Digit width varies between models, so
# both forms are listed wherever a numbered variant exists.
_BROW_SOFT = ("にこり", "にこり左", "にっこり")
_EYE_NARROW = ("なごみ", "なごみ左", "にっこり目")
_EYE_CLOSED = ("笑い", "笑い1", "笑")
_MOUTH_CORNERS = ("口角上げ", "口角上げ1", "口角上げ１")
_MOUTH_SMIRK = ("にやり", "にやり２", "にやり2", "にやり3", "にやり３")

EXPRESSION_PRESETS: dict[str, tuple[tuple[tuple[str, ...], float], ...]] = {
    # The default, and it deliberately leaves the eyes alone. Narrowing them does
    # read as a warmer smile in a close-up, but these characters are designed around
    # large open eyes: shrinking them changes who the character looks like, and at
    # full-body scale the irises vanish and it reads as a squint. Brows and mouth are
    # enough for a smile.
    "smile": (
        (_BROW_SOFT, 0.5),
        (_MOUTH_CORNERS, 0.75),
    ),
    # Roughly half strength, for a serious motion where a full smile would fight the
    # action. Eyes untouched, as above.
    "soft_smile": (
        (_BROW_SOFT, 0.3),
        (_MOUTH_CORNERS, 0.35),
    ),
    # A broad smile. Only for a shot where the mouth is large enough to survive it:
    # driven this hard the two strokes stretch and the centre gap opens up, which is
    # what makes it read as an open mouth at small sizes.
    "smile_wide": (
        (_BROW_SOFT, 0.5),
        (_MOUTH_CORNERS, 1.0),
        (_MOUTH_SMIRK, 0.6),
    ),
    # `smile` plus narrowed eyes. Warmer in a face close-up, but it costs the eye
    # shape, so it is opt-in rather than the default.
    "smile_eyes": (
        (_BROW_SOFT, 0.5),
        (_EYE_NARROW, 0.7),
        (_MOUTH_CORNERS, 0.75),
    ),
    # Eyes closed into arcs. Warmer, but the eyes are gone.
    "happy": (
        (_BROW_SOFT, 0.5),
        (_EYE_CLOSED, 1.0),
        (_MOUTH_CORNERS, 1.0),
        (_MOUTH_SMIRK, 0.6),
    ),
    # Barely-there pleasantness, for a serious or athletic motion where a smile
    # would fight the action.
    "calm": (
        (_BROW_SOFT, 0.3),
        (_MOUTH_CORNERS, 0.5),
    ),
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
