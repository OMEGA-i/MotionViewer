"""Root translation must be rescaled by leg length, not by height.

This had been the height ratio, which is right only when the two rigs share the same
leg-to-height proportion. Mixamo characters happen to (0.529 against SMPL-X's 0.528),
so the rule looked correct for as long as those were the only targets. A stylised MMD
character breaks it: Yoimiya is 0.7902 m of leg on 1.7208 m of height, a ratio of
0.4592 against the source's 0.5322, so the height rule asks the root to cover 15.9%
more ground than the legs can reach and the planted foot slides that fraction of every
metre travelled.

The regression these tests guard is therefore *not* a crash — it is a plausible number
that produces sliding proportional to distance, which reads as a retarget quality
problem rather than as a unit error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from motionviewer.blender.retarget._mmd_plan import _root_translation_scale  # noqa: E402


class _Vector(np.ndarray):
    """A numpy array that also answers to ``.z``, the way a Blender vector does."""

    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    @property
    def z(self) -> float:
        return float(self[2])


class _Matrix:
    """Just enough of a Blender matrix: a translation and left-multiplication."""

    def __init__(self, translation):
        self.translation = _Vector(translation)

    def __matmul__(self, other):
        return _Matrix(self.translation + other.translation)

    def to_scale(self):
        return (1.0, 1.0, 1.0)


class _Bone:
    def __init__(self, position, length=0.1):
        self.matrix_local = _Matrix(position)
        self.length = length


class _Armature:
    def __init__(self):
        self.matrix_world = _Matrix((0.0, 0.0, 0.0))


def _source_frame(position):
    frame = np.eye(4)
    frame[:3, 3] = position
    return frame


# Yoimiya's measured leg chain and the source skeleton's, both in metres.
TARGET_LEG = 0.7902
SOURCE_LEG = 0.8110
TARGET_HEIGHT = 1.7208
SOURCE_HEIGHT = 1.5239


def _rig(*, with_legs: bool = True):
    """A target rig and matching source rest, with the measured proportions."""
    target_to_source = {}
    bones: dict[str, _Bone] = {}
    source_rest: dict[str, np.ndarray] = {}

    if with_legs:
        for side in ("L", "R"):
            # Straight down the Z axis, so summed segment length is the leg length.
            bones[f"leg_{side}"] = _Bone((0.0, 0.0, TARGET_LEG))
            bones[f"knee_{side}"] = _Bone((0.0, 0.0, TARGET_LEG / 2.0))
            bones[f"foot_{side}"] = _Bone((0.0, 0.0, 0.0))
            target_to_source[f"leg_{side}"] = f"hip.{side}"
            target_to_source[f"knee_{side}"] = f"thigh.{side}"
            target_to_source[f"foot_{side}"] = f"foot.{side}"
            source_rest[f"hip.{side}"] = _source_frame((0.0, 0.0, SOURCE_LEG))
            source_rest[f"thigh.{side}"] = _source_frame((0.0, 0.0, SOURCE_LEG / 2.0))
            source_rest[f"foot.{side}"] = _source_frame((0.0, 0.0, 0.0))

    # A head far above the hips, so the height ratio differs from the leg ratio and the
    # two rules cannot be confused for one another. A floor-level bone comes with it:
    # height needs two mapped bones at different heights to be defined at all, and with
    # only one the function correctly reports a zero span and returns 1.0.
    bones["head"] = _Bone((0.0, 0.0, TARGET_HEIGHT), length=0.0)
    target_to_source["head"] = "head"
    source_rest["head"] = _source_frame((0.0, 0.0, SOURCE_HEIGHT))
    bones["ground"] = _Bone((0.0, 0.0, 0.0), length=0.0)
    target_to_source["ground"] = "ground"
    source_rest["ground"] = _source_frame((0.0, 0.0, 0.0))
    return _Armature(), bones, target_to_source, source_rest


def test_scale_is_the_leg_ratio():
    armature, bones, mapping, source_rest = _rig()
    scale = _root_translation_scale(armature, bones, mapping, source_rest, "head")
    assert scale == pytest.approx(TARGET_LEG / SOURCE_LEG, rel=1e-6)


def test_scale_is_not_the_height_ratio():
    """The old rule, kept as an explicit negative so a revert fails loudly."""
    armature, bones, mapping, source_rest = _rig()
    scale = _root_translation_scale(armature, bones, mapping, source_rest, "head")
    height_rule = TARGET_HEIGHT / SOURCE_HEIGHT
    assert abs(scale - height_rule) > 0.1


def test_the_two_rules_disagree_by_the_measured_amount():
    """15.9% on this rig — the fraction of every travelled metre the foot used to slide."""
    armature, bones, mapping, source_rest = _rig()
    leg_rule = _root_translation_scale(armature, bones, mapping, source_rest, "head")
    height_rule = TARGET_HEIGHT / SOURCE_HEIGHT
    assert (height_rule - leg_rule) / leg_rule == pytest.approx(0.159, abs=0.005)


def test_matching_proportions_make_the_rules_agree():
    """Why the height rule survived on Mixamo: there the proportions do match."""
    armature, bones, mapping, source_rest = _rig()
    # Put the head where the source's proportion would put it.
    bones["head"] = _Bone((0.0, 0.0, SOURCE_HEIGHT * TARGET_LEG / SOURCE_LEG), length=0.0)
    leg_rule = _root_translation_scale(armature, bones, mapping, source_rest, "head")
    height_rule = (SOURCE_HEIGHT * TARGET_LEG / SOURCE_LEG) / SOURCE_HEIGHT
    assert leg_rule == pytest.approx(height_rule, rel=1e-6)


def test_falls_back_to_height_when_no_leg_chain_resolves():
    """A rig without the leg roles must still rescale, not travel in source metres."""
    armature, bones, mapping, source_rest = _rig(with_legs=False)
    scale = _root_translation_scale(armature, bones, mapping, source_rest, "head")
    assert scale != pytest.approx(1.0)
    assert scale == pytest.approx(TARGET_HEIGHT / SOURCE_HEIGHT, rel=1e-6)


def test_an_empty_mapping_is_neutral_rather_than_an_error():
    armature = _Armature()
    assert _root_translation_scale(armature, {}, {}, {}, "head") == pytest.approx(1.0)
