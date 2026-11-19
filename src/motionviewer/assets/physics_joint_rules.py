"""Validated, asset-profile rules for secondary-physics joint compliance."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class JointRule:
    """Both dynamic endpoints must belong to this exact bone-name prefix.

    Root anchors and joints between different garment chains never match. Limits
    are solver targets, not guarantees of measured angular/positional error.
    """

    bone_prefix: str
    locked_angular_half_range_degrees: float = 0.0
    min_angular_half_range_degrees: float = 0.0
    locked_linear_half_range_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.bone_prefix, str) or not self.bone_prefix.strip():
            raise ValueError("joint rule bone_prefix must be nonempty")
        for name, high in (
            ("locked_angular_half_range_degrees", 45.0),
            ("min_angular_half_range_degrees", 45.0),
            ("locked_linear_half_range_m", 0.01),
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0.0 <= value <= high):
                raise ValueError(f"joint rule {name} must be finite and in [0, {high}]")
        if self.locked_angular_half_range_degrees and self.min_angular_half_range_degrees:
            raise ValueError("choose locked angular slack or a minimum angular range")
        if not any((self.locked_angular_half_range_degrees,
                    self.min_angular_half_range_degrees, self.locked_linear_half_range_m)):
            raise ValueError("joint rule must change at least one limit")

    def to_json(self) -> dict:
        return asdict(self)


def joint_rules_from_json(value: Any) -> tuple[JointRule, ...]:
    if not isinstance(value, list):
        raise ValueError("joint_rules must be a list")
    allowed = set(JointRule.__dataclass_fields__)
    rules = []
    for item in value:
        if (not isinstance(item, dict) or "bone_prefix" not in item
                or set(item) - allowed):
            raise ValueError("joint rule requires bone_prefix and known limit fields")
        rule = JointRule(**item)
        if any(rule.bone_prefix.startswith(old.bone_prefix)
               or old.bone_prefix.startswith(rule.bone_prefix) for old in rules):
            raise ValueError("joint rule prefixes must not overlap")
        rules.append(rule)
    return tuple(rules)


def widened_limits(lower: float, upper: float, half_range: float, *, locked_only: bool) -> tuple[float, float]:
    """Preserve wider/reversed limits; widen eligible ranges around their centre."""
    width = upper - lower
    if (half_range <= 0 or width < 0 or not all(map(math.isfinite, (lower, upper)))
            or (locked_only and width >= 1e-8) or width >= 2 * half_range):
        return lower, upper
    centre = (lower + upper) * 0.5
    return centre - half_range, centre + half_range


@dataclass(frozen=True)
class SideLinkRule:
    """Narrow the linear travel of joints that tie neighbouring garment strips together.

    A PMX coat built as a grid of columns has near-rigid joints down each column and
    looser *side* links between neighbouring columns. Furina's rear coat (``HB1``)
    allows 35-85 mm of sliding on its ``HB1横`` side links against 0 mm down a column,
    so each column swings as its own stick and the coat's silhouette splits into strips
    and snaps back together. Scaling the side links' linear range toward its centre makes
    neighbours travel together: at 0.3 the rate at which neighbouring strips separate fell
    from 8.5 to 3.9 mm/frame (p95).

    Unlike ``JointRule`` this only ever *narrows* limits, and it matches by joint name,
    because side and down-chain joints connect bodies with the same prefix.
    """

    joint_prefix: str
    linear_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.joint_prefix, str) or not self.joint_prefix.strip():
            raise ValueError("side link rule joint_prefix must be nonempty")
        value = self.linear_scale
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0.0 <= value < 1.0):
            raise ValueError("side link rule linear_scale must be finite and in [0, 1)")

    def to_json(self) -> dict:
        return asdict(self)


def side_link_rules_from_json(value: Any) -> tuple[SideLinkRule, ...]:
    if not isinstance(value, list):
        raise ValueError("side_link_rules must be a list")
    rules = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"joint_prefix", "linear_scale"}:
            raise ValueError("side link rule requires exactly joint_prefix and linear_scale")
        rule = SideLinkRule(**item)
        if any(rule.joint_prefix == old.joint_prefix for old in rules):
            raise ValueError("side link rule prefixes must be unique")
        rules.append(rule)
    return tuple(rules)


def narrowed_limits(lower: float, upper: float, scale: float) -> tuple[float, float]:
    """Scale a range toward its centre. Reversed or non-finite limits are kept."""
    if upper < lower or not all(map(math.isfinite, (lower, upper))):
        return lower, upper
    centre = (lower + upper) * 0.5
    return centre + (lower - centre) * scale, centre + (upper - centre) * scale
