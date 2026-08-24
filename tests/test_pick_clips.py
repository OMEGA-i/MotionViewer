"""Geometry and bookkeeping for the interactive clip picker.

The picker draws each bone as one instanced mesh oriented by a quaternion, so the
only thing that can be silently wrong is that rotation: a sign error still yields
unit quaternions and a plausible-looking skeleton, just with limbs pointing the
wrong way. These tests check the rotation actually maps +Y onto each bone, which is
the property the drawing relies on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from pick_clips import EDGES, _bone_mesh, _quat_from_y  # noqa: E402

from motionviewer.core.smplx_fk import SMPLX_BODY22_PARENTS  # noqa: E402


def _rotate_up(quaternions: np.ndarray) -> np.ndarray:
    """``R(q) @ (0, 1, 0)`` for unit quaternions in wxyz order."""
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    return np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_edges_cover_every_bone_once():
    assert len(EDGES) == sum(1 for parent in SMPLX_BODY22_PARENTS if parent >= 0) == 21
    assert {tuple(edge) for edge in EDGES} == {
        (parent, child) for child, parent in enumerate(SMPLX_BODY22_PARENTS) if parent >= 0
    }


def test_bone_mesh_is_a_closed_octahedron_along_positive_y():
    vertices, faces = _bone_mesh()
    assert vertices.shape == (6, 3) and faces.shape == (8, 3)
    # base at the origin, tip one unit up, so a per-bone scale is the bone's length
    assert np.allclose(vertices[0], (0.0, 0.0, 0.0))
    assert np.allclose(vertices[1], (0.0, 1.0, 0.0))
    # every vertex referenced, and each edge shared by exactly two faces
    assert set(faces.ravel().tolist()) == set(range(6))
    edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(face[a]), int(face[b]))))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}


@pytest.mark.parametrize(
    "direction",
    [
        (0.0, 1.0, 0.0),  # identity
        (0.0, -1.0, 0.0),  # half turn, the degenerate case
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.577, 0.577, 0.577),
    ],
)
def test_rotation_maps_up_onto_the_bone_direction(direction):
    target = _unit(np.array([direction], np.float32))
    assert _rotate_up(_quat_from_y(target)) == pytest.approx(target, abs=1e-5)


def test_rotation_holds_over_many_random_directions():
    rng = np.random.default_rng(0)
    target = _unit(rng.normal(size=(2000, 3)).astype(np.float32))
    assert _rotate_up(_quat_from_y(target)) == pytest.approx(target, abs=1e-5)


def test_quaternions_are_unit_length():
    rng = np.random.default_rng(1)
    target = _unit(rng.normal(size=(500, 3)).astype(np.float32))
    norms = np.linalg.norm(_quat_from_y(target), axis=1)
    assert norms == pytest.approx(np.ones(500), abs=1e-6)


def test_a_zero_direction_does_not_produce_nan():
    """A zero-length bone can occur if two joints coincide; it must not poison the frame."""
    quaternions = _quat_from_y(np.zeros((1, 3), np.float32))
    assert np.isfinite(quaternions).all()


# ---------------------------------------------------------------------------
# shortlist and tag persistence
# ---------------------------------------------------------------------------

from pick_clips import (  # noqa: E402
    picks_file,
    read_picks,
    read_tags,
    tags_file,
    write_shortlist,
    write_tags,
)

META = {"000015": ("a person throws a ball", 40), "028509": ("a person dances", 82)}


def test_shortlist_is_shaped_for_the_clips_flag(tmp_path):
    """The .txt is pasted straight into --clips, so it must be exactly that format."""
    write_shortlist(tmp_path, "live500/gen", ["028509", "000015"], {}, META)
    text = picks_file(tmp_path, "live500/gen").read_text(encoding="utf-8")
    assert text.strip() == "000015,028509"
    assert picks_file(tmp_path, "live500/gen").name == "live500-gen.txt"


def test_shortlist_deduplicates_and_sorts(tmp_path):
    write_shortlist(tmp_path, "t/gen", ["028509", "000015", "028509"], {}, META)
    assert read_picks(tmp_path, "t/gen") == ["000015", "028509"]


def test_an_empty_shortlist_writes_an_empty_file(tmp_path):
    write_shortlist(tmp_path, "t/gen", [], {}, META)
    assert picks_file(tmp_path, "t/gen").read_text(encoding="utf-8") == ""
    assert read_picks(tmp_path, "t/gen") == []


def test_shortlist_json_carries_tag_and_caption(tmp_path):
    write_shortlist(tmp_path, "t/gen", ["000015"], {"000015": "big wind-up, clean landing"}, META)
    payload = json.loads(picks_file(tmp_path, "t/gen").with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["task"] == "t/gen"
    entry = payload["clips"][0]
    assert entry["tag"] == "big wind-up, clean landing"
    assert entry["caption"] == "a person throws a ball"
    assert entry["frames"] == 40


def test_tags_round_trip(tmp_path):
    write_tags(tmp_path, "t/gen", {"000015": " throw ", "028509": "dance"})
    assert read_tags(tmp_path, "t/gen") == {"000015": "throw", "028509": "dance"}


def test_clearing_a_tag_removes_it(tmp_path):
    write_tags(tmp_path, "t/gen", {"000015": "throw"})
    write_tags(tmp_path, "t/gen", {"000015": "   "})
    assert read_tags(tmp_path, "t/gen") == {}


def test_tags_survive_un_keeping(tmp_path):
    """Tags are a separate file for exactly this: dropping a pick keeps the note."""
    write_tags(tmp_path, "t/gen", {"000015": "throw"})
    write_shortlist(tmp_path, "t/gen", ["000015"], read_tags(tmp_path, "t/gen"), META)
    write_shortlist(tmp_path, "t/gen", [], read_tags(tmp_path, "t/gen"), META)
    assert read_tags(tmp_path, "t/gen") == {"000015": "throw"}


def test_missing_files_read_as_empty(tmp_path):
    assert read_picks(tmp_path, "nope/gen") == []
    assert read_tags(tmp_path, "nope/gen") == {}


def test_a_corrupt_tag_file_does_not_crash_the_session(tmp_path):
    tags_file(tmp_path, "t/gen").write_text("{not json", encoding="utf-8")
    assert read_tags(tmp_path, "t/gen") == {}


def test_unicode_tags_survive(tmp_path):
    write_tags(tmp_path, "t/gen", {"000015": "投球 · 起势大 · 落地干净"})
    assert read_tags(tmp_path, "t/gen")["000015"] == "投球 · 起势大 · 落地干净"
