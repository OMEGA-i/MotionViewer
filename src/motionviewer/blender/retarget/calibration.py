"""Mixamo rig identification, preflight validation, and import calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

# These are deliberately unprefixed.  A ``MixamoNameAdapter`` owns the sole
# conversion to an FBX-specific namespace.
MIXAMO_BODY_BONES: tuple[str, ...] = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)

MIXAMO_PARENT_BONES: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}

MIXAMO_PREFIXES: tuple[str, ...] = ("", "mixamorig:", "mixamorig1:")
MIN_MIXAMO_HEIGHT_M = 0.25
MAX_MIXAMO_HEIGHT_M = 4.0

_JOINT_LIMIT_DEGREES: dict[str, tuple[float, float]] = {
    "Hips": (180.0, 180.0),
    "Spine": (45.0, 35.0),
    "Spine1": (45.0, 35.0),
    "Spine2": (60.0, 45.0),
    "Neck": (75.0, 60.0),
    "Head": (90.0, 75.0),
    "LeftShoulder": (100.0, 90.0),
    "RightShoulder": (100.0, 90.0),
    "LeftArm": (180.0, 180.0),
    "RightArm": (180.0, 180.0),
    "LeftForeArm": (170.0, 170.0),
    "RightForeArm": (170.0, 170.0),
    "LeftHand": (120.0, 120.0),
    "RightHand": (120.0, 120.0),
    "LeftUpLeg": (170.0, 170.0),
    "RightUpLeg": (170.0, 170.0),
    "LeftLeg": (170.0, 170.0),
    "RightLeg": (170.0, 170.0),
    "LeftFoot": (100.0, 90.0),
    "RightFoot": (100.0, 90.0),
    "LeftToeBase": (70.0, 45.0),
    "RightToeBase": (70.0, 45.0),
}


@dataclass(frozen=True)
class MixamoNameAdapter:
    """Maps the canonical Mixamo bone namespace to an imported FBX rig."""

    prefix: str

    @classmethod
    def detect(cls, bone_names: set[str]) -> MixamoNameAdapter | None:
        # Hips is mandatory and makes prefix selection deterministic.  Do not
        # use fuzzy matching: a partial or hybrid rig must fail preflight.
        matches = [prefix for prefix in MIXAMO_PREFIXES if f"{prefix}Hips" in bone_names]
        return cls(prefix=matches[0]) if len(matches) == 1 else None

    def target_name(self, canonical_name: str) -> str:
        if canonical_name not in MIXAMO_BODY_BONES:
            raise KeyError(f"Unknown Mixamo canonical bone {canonical_name!r}")
        return f"{self.prefix}{canonical_name}"

    def target_map(self) -> dict[str, str]:
        return {name: self.target_name(name) for name in MIXAMO_BODY_BONES}


@dataclass(frozen=True)
class MixamoRigInspection:
    """Deterministic structural checks performed before retargeting."""

    adapter: MixamoNameAdapter | None
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.adapter is not None and not self.errors


@dataclass(frozen=True)
class CalibrationResult:
    rig_family: str
    prefix: str | None
    original_rotation: tuple[float, float, float]
    original_scale: tuple[float, float, float]
    object_basis: tuple[tuple[float, float, float, float], ...]
    uniform_scale: float
    unit_correction: float
    transform_baked: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "rig_family": self.rig_family,
            "prefix": self.prefix,
            "original_rotation": list(self.original_rotation),
            "original_scale": list(self.original_scale),
            "object_basis": [list(row) for row in self.object_basis],
            "uniform_scale": self.uniform_scale,
            "unit_correction": self.unit_correction,
            "transform_baked": self.transform_baked,
        }


@dataclass(frozen=True)
class MixamoRigProfile:
    """Persistable evidence for one approved imported Mixamo asset."""

    asset_path: str
    asset_sha256: str | None
    armature_name: str
    prefix: str
    root_bone: str
    bone_map: dict[str, str]
    bone_lengths: dict[str, float]
    height_m: float
    parent_map: dict[str, str | None]
    rest_matrices: dict[str, list[list[float]]]
    local_axes: dict[str, tuple[float, float, float]]
    action_name: str | None
    calibration: CalibrationResult
    sole_support_points: dict[str, dict[str, Any]]
    generated_at: str
    blender_version: str | None
    mirror_pairs: dict[str, str]
    joint_limits: dict[str, dict[str, float]]
    rotation_offsets_wxyz: dict[str, tuple[float, float, float, float]]
    validation_errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.validation_errors

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "motionviewer.mixamo_rig.v2",
            "asset_path": self.asset_path,
            "asset_sha256": self.asset_sha256,
            "armature_name": self.armature_name,
            "prefix": self.prefix,
            "root_bone": self.root_bone,
            "bone_map": self.bone_map,
            "bone_lengths": self.bone_lengths,
            "height_m": self.height_m,
            "parent_map": self.parent_map,
            "rest_matrices": self.rest_matrices,
            "local_axes": {name: list(axis) for name, axis in self.local_axes.items()},
            "action_name": self.action_name,
            "calibration": self.calibration.to_json(),
            "sole_support_points": self.sole_support_points,
            "generated_at": self.generated_at,
            "blender_version": self.blender_version,
            "mirror_pairs": self.mirror_pairs,
            "joint_limits": self.joint_limits,
            "rotation_offsets_wxyz": {
                name: list(value) for name, value in self.rotation_offsets_wxyz.items()
            },
            "validation_errors": list(self.validation_errors),
        }


def inspect_mixamo_rig(fbx_armature: Any) -> MixamoRigInspection:
    bones = {bone.name: bone for bone in fbx_armature.data.bones}
    adapter = MixamoNameAdapter.detect(set(bones))
    if adapter is None:
        return MixamoRigInspection(None, ("missing Mixamo Hips root",))

    errors: list[str] = []
    for canonical in MIXAMO_BODY_BONES:
        name = adapter.target_name(canonical)
        bone = bones.get(name)
        if bone is None:
            errors.append(f"missing Mixamo bone: {canonical}")
            continue
        expected_parent = MIXAMO_PARENT_BONES[canonical]
        actual_parent = getattr(getattr(bone, "parent", None), "name", None)
        expected_actual = adapter.target_name(expected_parent) if expected_parent else None
        if actual_parent != expected_actual:
            errors.append(
                f"invalid parent for {canonical}: expected {expected_actual!r}, got {actual_parent!r}"
            )
        length = getattr(bone, "length", None)
        if length is not None and float(length) <= 1e-7:
            errors.append(f"degenerate Mixamo bone: {canonical}")
    return MixamoRigInspection(adapter, tuple(errors))


def build_mixamo_rig_profile(
    fbx_path: str | Path,
    fbx_armature: Any,
    calibration: CalibrationResult,
    fbx_meshes: list[Any] | None = None,
) -> MixamoRigProfile:
    """Capture the imported rig after transforms have been normalized."""
    inspection = inspect_mixamo_rig(fbx_armature)
    if inspection.adapter is None:
        raise ValueError("Imported FBX is not a supported Mixamo rig: " + "; ".join(inspection.errors))
    adapter = inspection.adapter
    path = Path(fbx_path)
    digest = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    bones = {bone.name: bone for bone in fbx_armature.data.bones}
    bone_map = adapter.target_map()

    def rest_matrix(name: str) -> list[list[float]]:
        matrix = getattr(bones[name], "matrix_local", None)
        if matrix is None:
            return []
        return _rigid_matrix_rows(fbx_armature.matrix_world @ matrix)

    def local_axis(name: str) -> tuple[float, float, float]:
        matrix = getattr(bones[name], "matrix_local", None)
        if matrix is None:
            return (0.0, 1.0, 0.0)
        column = matrix.to_3x3().col[1]
        return (float(column.x), float(column.y), float(column.z))

    action = getattr(getattr(fbx_armature, "animation_data", None), "action", None)
    rest_matrices = {canonical: rest_matrix(name) for canonical, name in bone_map.items()}
    bone_lengths = {
        canonical: float(
            (
                fbx_armature.matrix_world @ bones[name].tail_local
                - fbx_armature.matrix_world @ bones[name].head_local
            ).length
        )
        for canonical, name in bone_map.items()
    }
    height_m = _mixamo_world_height(fbx_armature, adapter)
    validation_errors = list(inspection.errors)
    if not np.isfinite(height_m) or not MIN_MIXAMO_HEIGHT_M <= height_m <= MAX_MIXAMO_HEIGHT_M:
        validation_errors.append(
            "Mixamo world height is outside the supported range "
            f"[{MIN_MIXAMO_HEIGHT_M:g}, {MAX_MIXAMO_HEIGHT_M:g}] m: {height_m:g} m"
        )
    return MixamoRigProfile(
        asset_path=str(path),
        asset_sha256=digest,
        armature_name=str(fbx_armature.name),
        prefix=adapter.prefix,
        root_bone=adapter.target_name("Hips"),
        bone_map=bone_map,
        bone_lengths=bone_lengths,
        height_m=height_m,
        parent_map={
            canonical: (
                None
                if MIXAMO_PARENT_BONES[canonical] is None
                else adapter.target_name(MIXAMO_PARENT_BONES[canonical])
            )
            for canonical in MIXAMO_BODY_BONES
        },
        rest_matrices=rest_matrices,
        local_axes={canonical: local_axis(name) for canonical, name in bone_map.items()},
        action_name=None if action is None else str(action.name),
        calibration=calibration,
        sole_support_points=extract_mixamo_sole_anchors(fbx_armature, fbx_meshes or [], adapter=adapter),
        generated_at=datetime.now(UTC).isoformat(),
        blender_version=_blender_version(),
        mirror_pairs={
            name: ("Right" + name[4:] if name.startswith("Left") else "Left" + name[5:])
            for name in MIXAMO_BODY_BONES
            if name.startswith(("Left", "Right"))
        },
        joint_limits={
            name: {
                "maximum_swing_degrees": limits[0],
                "maximum_twist_degrees": limits[1],
                "maximum_rotation_degrees": max(limits),
            }
            for name, limits in _JOINT_LIMIT_DEGREES.items()
        },
        rotation_offsets_wxyz={name: (1.0, 0.0, 0.0, 0.0) for name in MIXAMO_BODY_BONES},
        validation_errors=tuple(validation_errors),
    )


def _blender_version() -> str | None:
    try:
        import bpy  # type: ignore
    except ImportError:
        return None
    return str(bpy.app.version_string)


def _rigid_matrix_rows(matrix: Any) -> list[list[float]]:
    """Return a scale-free affine matrix while preserving world translation."""
    rigid = matrix.to_quaternion().to_matrix().to_4x4()
    rigid.translation = matrix.translation
    return [[float(value) for value in row] for row in rigid]


def extract_mixamo_sole_anchors(
    fbx_armature: Any,
    fbx_meshes: list[Any],
    *,
    adapter: MixamoNameAdapter | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure compact sole support points from Mixamo foot vertex weights."""
    if adapter is None:
        adapter = MixamoNameAdapter.detect({bone.name for bone in fbx_armature.data.bones})
    if adapter is None:
        return {}
    bones = {bone.name: bone for bone in fbx_armature.data.bones}
    result: dict[str, dict[str, Any]] = {}
    for side, contact_name in (("Left", "left_foot"), ("Right", "right_foot")):
        foot_name = adapter.target_name(f"{side}Foot")
        toe_name = adapter.target_name(f"{side}ToeBase")
        foot_bone = bones.get(foot_name)
        if foot_bone is None:
            continue
        candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        bone_world = fbx_armature.matrix_world @ foot_bone.matrix_local
        raw_world_to_bone = bone_world.inverted()
        rigid_bone_world = bone_world.to_quaternion().to_matrix().to_4x4()
        rigid_bone_world.translation = bone_world.translation
        world_to_rigid_bone = rigid_bone_world.inverted()
        for mesh in fbx_meshes:
            group_indices = {
                int(group.index)
                for group in getattr(mesh, "vertex_groups", ())
                if group.name in {foot_name, toe_name}
            }
            if not group_indices:
                continue
            for vertex in getattr(getattr(mesh, "data", None), "vertices", ()):
                influence = sum(
                    float(member.weight)
                    for member in getattr(vertex, "groups", ())
                    if int(member.group) in group_indices
                )
                if influence < 0.2:
                    continue
                world = mesh.matrix_world @ vertex.co
                local_m = world_to_rigid_bone @ world
                bone_local = raw_world_to_bone @ world
                candidates.append(
                    (
                        np.asarray(tuple(float(value) for value in world), dtype=np.float64),
                        np.asarray(tuple(float(value) for value in local_m), dtype=np.float64),
                        np.asarray(tuple(float(value) for value in bone_local), dtype=np.float64),
                    )
                )
        if not candidates:
            continue
        world_points = np.stack([item[0] for item in candidates])
        local_points_m = np.stack([item[1] for item in candidates])
        bone_local_points = np.stack([item[2] for item in candidates])
        foot_span = float(getattr(foot_bone, "length", 0.0)) + float(
            getattr(bones.get(toe_name), "length", 0.0)
        )
        support_band = max(0.005, 0.1 * foot_span)
        low_indices = np.flatnonzero(world_points[:, 2] <= float(np.min(world_points[:, 2])) + support_band)
        low_local = local_points_m[low_indices]
        # Keep a stable, labelled support quadrilateral.  Passing an arbitrary
        # cloud of skin vertices to an IK solver makes its plane/heading depend
        # on topology density instead of the actual sole geometry.
        extrema = {
            "heel": int(low_indices[int(np.argmin(low_local[:, 1]))]),
            "toe": int(low_indices[int(np.argmax(low_local[:, 1]))]),
            "medial": int(low_indices[int(np.argmin(low_local[:, 0]))]),
            "lateral": int(low_indices[int(np.argmax(low_local[:, 0]))]),
        }
        selected = list(dict.fromkeys(extrema.values()))
        if len(selected) < 4:
            remaining = [int(index) for index in low_indices if int(index) not in selected]
            centroid = np.mean(low_local[:, :2], axis=0)
            remaining.sort(
                key=lambda index: float(np.linalg.norm(local_points_m[index, :2] - centroid)),
                reverse=True,
            )
            selected.extend(remaining[: 4 - len(selected)])
        selected = selected[:4]
        points = np.stack([local_points_m[index] for index in selected])
        centered = points - np.mean(points, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        forward = points[1] - points[0] if len(points) >= 2 else np.array((0.0, 1.0, 0.0))
        forward -= normal * float(np.dot(forward, normal))
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        lateral = np.cross(forward, normal)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        normal = np.cross(lateral, forward)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        # SVD plane normals have an arbitrary sign.  A downward semantic
        # normal makes the contact solver rotate a geometrically upright foot
        # by almost 180 degrees when it locks the sole to world +Z.  Resolve
        # that ambiguity in calibrated rest-world space and then rebuild the
        # complete right-handed sole frame from the semantic heel-toe axis.
        rest_normal_world = np.asarray(rigid_bone_world.to_3x3(), dtype=np.float64) @ normal
        if rest_normal_world[2] < 0.0:
            normal *= -1.0
        lateral = np.cross(forward, normal)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        normal = np.cross(lateral, forward)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        planar_residual = float(singular_values[-1] / max(singular_values[0], 1e-12))
        result[contact_name] = {
            "bone": foot_name,
            "anchor_labels": [name for name, index in extrema.items() if index in selected],
            "points_local": [bone_local_points[index].tolist() for index in selected],
            "points_local_m": [local_points_m[index].tolist() for index in selected],
            "sole_forward_local": forward.tolist(),
            "sole_normal_local": normal.tolist(),
            "sole_lateral_local": lateral.tolist(),
            "planar_residual": planar_residual,
        }
    return result


def detect_mixamo_family(fbx_armature: Any) -> str:
    """Return the single supported rig family, independent of namespace."""
    return (
        "mixamo" if MixamoNameAdapter.detect({bone.name for bone in fbx_armature.data.bones}) else "unknown"
    )


def infer_mixamo_unit_correction(height_m: float) -> float:
    """Infer only decimal unit mismatches from an implausible Mixamo stature."""
    height = float(height_m)
    if not np.isfinite(height) or height <= 0.0:
        return 1.0
    if MIN_MIXAMO_HEIGHT_M <= height <= MAX_MIXAMO_HEIGHT_M:
        return 1.0
    candidates = [10.0**exponent for exponent in range(-6, 7)]
    plausible = [
        value for value in candidates if MIN_MIXAMO_HEIGHT_M <= height * value <= MAX_MIXAMO_HEIGHT_M
    ]
    if not plausible:
        return 1.0
    reference_height_m = 1.7
    return min(
        plausible,
        key=lambda value: abs(np.log10(height * value / reference_height_m)),
    )


def _mixamo_world_height(fbx_armature: Any, adapter: MixamoNameAdapter | None = None) -> float:
    if adapter is None:
        adapter = MixamoNameAdapter.detect({bone.name for bone in fbx_armature.data.bones})
    if adapter is None:
        return float("nan")
    points = []
    for canonical in MIXAMO_BODY_BONES:
        bone = fbx_armature.data.bones.get(adapter.target_name(canonical))
        if bone is None:
            continue
        points.extend(
            (fbx_armature.matrix_world @ bone.head_local, fbx_armature.matrix_world @ bone.tail_local)
        )
    if not points:
        return float("nan")
    z_values = [float(point.z) for point in points]
    return max(z_values) - min(z_values)


def normalize_imported_mixamo_units(fbx_armature: Any, fbx_meshes: list[Any] | None = None) -> float:
    """Normalize a bad FBX length declaration without baking object transforms."""
    correction = infer_mixamo_unit_correction(_mixamo_world_height(fbx_armature))
    if correction == 1.0:
        return correction

    fbx_armature.scale = tuple(float(value) * correction for value in fbx_armature.scale)

    # A conventional Mixamo mesh is parented to the armature and inherits the
    # correction.  Scale only independent mesh roots so every imported layout
    # remains coherent without applying transforms to bones or vertex data.
    for mesh in fbx_meshes or []:
        parent = getattr(mesh, "parent", None)
        while parent is not None and parent is not fbx_armature:
            parent = getattr(parent, "parent", None)
        if parent is fbx_armature:
            continue
        mesh.scale = tuple(float(value) * correction for value in mesh.scale)
        mesh.location = tuple(float(value) * correction for value in mesh.location)
    return correction


def calibrate_imported_fbx(
    fbx_armature: Any,
    *,
    unit_correction: float = 1.0,
    original_scale: tuple[float, float, float] | None = None,
) -> CalibrationResult:
    """Validate and record an imported Mixamo object basis without mutating it."""
    adapter = MixamoNameAdapter.detect({bone.name for bone in fbx_armature.data.bones})
    if adapter is None:
        raise ValueError("Unsupported FBX rig: expected a complete Mixamo skeleton")
    original_rotation = tuple(float(value) for value in fbx_armature.rotation_euler)
    imported_scale = (
        tuple(float(value) for value in fbx_armature.scale)
        if original_scale is None
        else tuple(float(value) for value in original_scale)
    )
    current_scale = tuple(float(value) for value in fbx_armature.scale)
    if min(current_scale) <= 1e-8:
        raise ValueError("Unsupported Mixamo object basis: degenerate or mirrored scale")
    local_uniform_scale = float(sum(current_scale) / 3.0)
    relative_spread = max(abs(value - local_uniform_scale) for value in current_scale) / local_uniform_scale
    if relative_spread > 1e-4:
        raise ValueError(
            "Unsupported Mixamo object basis: non-uniform scale "
            f"{current_scale!r}; re-export the FBX with uniform units"
        )
    object_basis = tuple(tuple(float(value) for value in row) for row in fbx_armature.matrix_world)
    rotation_scale = np.asarray([row[:3] for row in object_basis[:3]], dtype=np.float64)
    axis_scales = np.linalg.norm(rotation_scale, axis=0)
    uniform_scale = float(np.mean(axis_scales))
    world_spread = float(np.max(np.abs(axis_scales - uniform_scale)) / uniform_scale)
    if world_spread > 1e-4:
        raise ValueError("Unsupported Mixamo world basis: non-uniform parent scale")
    determinant = float(np.linalg.det(rotation_scale / uniform_scale))
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-3:
        raise ValueError("Unsupported Mixamo object basis: shear or mirrored rotation")
    return CalibrationResult(
        rig_family="mixamo",
        prefix=adapter.prefix,
        original_rotation=original_rotation,
        original_scale=imported_scale,
        object_basis=object_basis,
        uniform_scale=uniform_scale,
        unit_correction=float(unit_correction),
        transform_baked=False,
    )
