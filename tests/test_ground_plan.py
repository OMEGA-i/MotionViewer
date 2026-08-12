"""Regression tests for pure ground-plan calculations."""

from __future__ import annotations

from motionviewer.blender.retarget._ground import vertical_offset_from_heights


def test_negative_mesh_hang_raises_target_instead_of_lowering_it() -> None:
    """Soles below a foot bone need an upward, not downward, correction."""
    assert vertical_offset_from_heights(0.0, 0.0, -0.0363) == 0.0363


def test_vertical_offset_aligns_the_lowest_mesh_point_to_source_foot_height() -> None:
    offset = vertical_offset_from_heights(0.12, 0.20, -0.05)
    assert abs((0.20 - 0.05 + offset) - 0.12) < 1e-9
