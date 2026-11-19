"""Optional box-specific hand skin fitting and local arm IK on a derived actor.

Original motion files and root trajectory are never edited. The fixed box animation
is the target; only shoulder/elbow/wrist and finger rotations are baked here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from motionviewer.core.prop_fit import box_signed_distance, box_triangle_intersections


def audit_animated_box_clearance(bpy: Any, box: Any, meshes: list[Any], *,
                                size: tuple[float, float, float], frame_start: int,
                                frames: int, evaluation_stage: str) -> dict:
    """Read evaluated actor triangles at every animation frame without changing keys.

    This can run after Bullet baking, unlike the builder's pre-physics report.
    Triangle intersections are exact at each sampled frame; penetration depths
    use a 45-point barycentric grid and are lower bounds on the true maximum.
    """
    size_array = np.asarray(size, dtype=float)
    inset = .0002
    scene = bpy.context.scene
    original_frame = scene.frame_current
    rows = []
    barycentric = np.array([(u / 8, v / 8) for u in range(9) for v in range(9 - u)])
    try:
        for frame in range(frame_start, frame_start + frames):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            inverse = np.linalg.inv(np.asarray(box.matrix_world))
            depsgraph = bpy.context.evaluated_depsgraph_get()
            count, depth = 0, 0.0
            for mesh in meshes:
                obj = mesh.evaluated_get(depsgraph)
                data = obj.to_mesh()
                try:
                    data.calc_loop_triangles()
                    if not data.loop_triangles:
                        continue
                    points = np.empty((len(data.vertices), 3))
                    data.vertices.foreach_get("co", points.ravel())
                    matrix = inverse @ np.asarray(obj.matrix_world)
                    points = points @ matrix[:3, :3].T + matrix[:3, 3]
                    indices = np.asarray([tuple(t.vertices) for t in data.loop_triangles], int)
                    triangles = points[indices]
                    near = np.all(triangles.min(axis=1) < size_array / 2, axis=1)
                    near &= np.all(triangles.max(axis=1) > -size_array / 2, axis=1)
                    triangles = triangles[near]
                    hits = box_triangle_intersections(triangles, size_array, inset=inset)
                    triangles = triangles[hits]
                    count += len(triangles)
                    if len(triangles):
                        samples = (triangles[:, None, 0]
                                   + barycentric[None, :, 0, None] * (triangles[:, None, 1] - triangles[:, None, 0])
                                   + barycentric[None, :, 1, None] * (triangles[:, None, 2] - triangles[:, None, 0]))
                        depth = max(depth, -float(box_signed_distance(samples.reshape(-1, 3), size_array).min()) * 1000)
                finally:
                    obj.to_mesh_clear()
            rows.append({"frame": frame, "actor_triangle_intersections": count,
                         "sampled_penetration_max_mm": depth})
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()
    return {"evaluation_stage": evaluation_stage, "frame_count": frames,
            "box_inset_mm": inset * 1000,
            "actor_triangle_intersections_max": max((r["actor_triangle_intersections"] for r in rows), default=0),
            "frames_with_intersections": [r["frame"] for r in rows if r["actor_triangle_intersections"]],
            "sampled_penetration_max_mm": max((r["sampled_penetration_max_mm"] for r in rows), default=0.0),
            "frames": rows,
            "limitations": "Discrete animation frames only; no continuous-time collision check. "
                           "Depth samples are lower bounds on exact penetration depth."}


def _mesh_selections(armature: Any, meshes: list[Any]) -> list[dict]:
    selected = []
    for mesh in meshes:
        mesh.data.calc_loop_triangles()
        triangles = np.asarray([tuple(triangle.vertices) for triangle in mesh.data.loop_triangles], int)
        groups = {group.index: group.name for group in mesh.vertex_groups}
        weights = [{groups[g.group]: g.weight for g in vertex.groups if g.group in groups}
                   for vertex in mesh.data.vertices]
        hands = []
        for side in ("左", "右"):
            names = {name for name in groups.values() if name.startswith(side)
                     and any(part in name for part in ("手首", "人指", "中指", "薬指", "小指", "親指"))}
            ids = np.array([i for i, w in enumerate(weights) if sum(w.get(name, 0) for name in names) >= .7], int)
            palm = np.array([i for i in ids if weights[i].get(side + "手首", 0) >= .75], int)
            tips = {}
            for finger, end in (("人指", "３"), ("中指", "３"), ("薬指", "３"), ("小指", "３"), ("親指", "２")):
                tips[finger] = np.array([i for i in ids if weights[i].get(side + finger + end, 0) >= .5], int)
            hand_triangles = triangles[np.all(np.isin(triangles, ids), axis=1)]
            upper = armature.data.bones.get(side + "腕")
            limb_names = {bone.name for bone in upper.children_recursive} | {upper.name} if upper else names
            limb_ids = np.array([i for i, w in enumerate(weights) if sum(w.get(name, 0) for name in limb_names) >= .7], int)
            hands.append({"all": ids, "palm": palm, "tips": tips, "triangles": hand_triangles,
                          "limb": limb_ids})
        body_names = {"頭", "首", "上半身", "上半身2", "下半身", "腰"}
        body = np.array([i for i, w in enumerate(weights) if sum(w.get(name, 0) for name in body_names) >= .7], int)
        body_triangles = triangles[np.all(np.isin(triangles, body), axis=1)]
        selected.append({"mesh": mesh, "hands": hands, "body": body, "triangles": triangles,
                         "body_triangles": body_triangles})
    return selected


def _skin_points(bpy: Any, selections: list[dict], box_matrix: np.ndarray) -> list[tuple[dict, np.ndarray]]:
    result = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    inverse = np.linalg.inv(box_matrix)
    for selection in selections:
        obj = selection["mesh"].evaluated_get(depsgraph)
        data = obj.to_mesh()
        try:
            points = np.empty((len(data.vertices), 3))
            data.vertices.foreach_get("co", points.ravel())
            transform = inverse @ np.asarray(obj.matrix_world)
            points = points @ transform[:3, :3].T + transform[:3, 3]
            result.append((selection, points))
        finally:
            obj.to_mesh_clear()
    return result


def _hand_points(skin, side_index: int, key: str, finger: str | None = None):
    chunks = []
    for selection, coords in skin:
        ids = selection["hands"][side_index][key]
        if finger is not None:
            ids = ids[finger]
        if len(ids):
            chunks.append(coords[ids])
            if key == "all":
                triangles = coords[selection["hands"][side_index]["triangles"]]
                chunks.append(triangles.mean(axis=1))
                chunks.append(((triangles + np.roll(triangles, -1, axis=1)) / 2).reshape(-1, 3))
    return np.concatenate(chunks) if chunks else np.empty((0, 3))


def _world_matrix(bpy: Any, armature: Any, bone: Any, rotation: np.ndarray, location: np.ndarray | None = None):
    from mathutils import Matrix, Vector

    world = armature.matrix_world @ bone.matrix
    desired = Matrix(rotation.tolist()).to_4x4()
    desired.translation = Vector(location) if location is not None else world.translation
    bone.matrix = armature.matrix_world.inverted() @ desired
    bpy.context.view_layer.update()


def _degrees_between(before, after) -> float:
    delta = before.transposed() @ after
    return float(np.degrees(delta.to_quaternion().angle))


def _calibrate_hand(bpy: Any, armature: Any, selections: list[dict], box: Any,
                    size: np.ndarray, prefix: str, side_index: int) -> dict:
    """Fit one hand against actual evaluated skin, then reuse that rigid template."""
    from mathutils import Euler

    from motionviewer.core.limb_ik import solve_two_bone_positions

    from .mmd_contact import _set_world_rotation
    from .retarget.mmd_solve import rotation_between

    bones = armature.pose.bones
    hand = bones[prefix + "手首"]
    names = [name for name in bones.keys() if name.startswith(prefix)
             and any(part in name for part in ("人指", "中指", "薬指", "小指", "親指"))]
    upper, lower = bones[prefix + "腕"], bones[prefix + "ひじ"]
    saved = {name: bones[name].matrix_basis.copy() for name in [upper.name, lower.name, hand.name, *names]}
    box_world = np.asarray(box.matrix_world).copy()
    rotation = box_world[:3, :3]
    hand_world = np.asarray(armature.matrix_world @ hand.matrix).copy()
    wrist = hand_world[:3, 3]
    middle = bones[prefix + "中指１"]
    forward = np.asarray(armature.matrix_world @ middle.head) - wrist
    forward /= np.linalg.norm(forward)
    tip = bones[prefix + "中指３"]
    before_tip = np.asarray(armature.matrix_world @ tip.tail).copy()
    for segment in "１２３":
        bone = bones[prefix + "中指" + segment]
        bone.rotation_quaternion = bone.rotation_quaternion @ Euler((.2, 0, 0), "XYZ").to_quaternion()
    bpy.context.view_layer.update()
    normal = np.asarray(armature.matrix_world @ tip.tail) - before_tip
    normal -= forward * np.dot(normal, forward)
    normal /= np.linalg.norm(normal)
    for name, basis in saved.items():
        bones[name].matrix_basis = basis
    bpy.context.view_layer.update()
    source_basis = np.column_stack((forward, normal, np.cross(forward, normal)))
    sign = float(np.sign((wrist - box_world[:3, 3]) @ rotation[:, 0]))
    target_forward = rotation[:, 2]
    target_normal = -sign * rotation[:, 0] + .65 * rotation[:, 1]
    target_normal /= np.linalg.norm(target_normal)
    target_basis = np.column_stack((target_forward, target_normal, np.cross(target_forward, target_normal)))
    wrist_rotation = target_basis @ source_basis.T @ hand_world[:3, :3]
    local_rotation = rotation.T @ wrist_rotation
    # Grip the rear upper corners: putting the whole wrist inside the box's depth
    # makes the approaching forearm/cuff cross its side even when fingers are clear.
    position = np.array((sign * (size[0] / 2 + .016), -size[1] / 2 - .030, size[2] / 2 - .09))
    fingers = {}
    finger_spread = {}
    references = {}
    for finger in ("人指", "中指", "薬指", "小指"):
        chain = [bones[prefix + finger + segment] for segment in "１２３"]
        references[finger] = np.array([tuple(b.rotation_quaternion.to_euler("XYZ")) for b in chain])
        fingers[finger] = np.array((5., 80., 20.))
        finger_spread[finger] = 0.0
    thumb_names = [prefix + "親指" + segment for segment in "０１２" if prefix + "親指" + segment in bones]
    thumb_reference = np.array([tuple(bones[name].rotation_quaternion.to_euler("XYZ")) for name in thumb_names])
    thumb_delta = np.zeros_like(thumb_reference)
    original_points = [np.asarray(armature.matrix_world @ bone.head).copy() for bone in (upper, lower, hand)]
    original_upper_rotation = np.asarray((armature.matrix_world @ upper.matrix).to_quaternion().to_matrix()).copy()

    def apply():
        from mathutils import Matrix, Quaternion

        # Calibration must bend the arm to the trial wrist, too. Translating only
        # the wrist distorts blended wrist/forearm skin and gives a false clearance.
        for bone in (upper, lower, hand):
            bone.matrix_basis = saved[bone.name]
        bpy.context.view_layer.update()
        shoulder, elbow, old_wrist = original_points
        goal = box_world[:3, 3] + rotation @ position
        middle, reached, _ = solve_two_bone_positions(shoulder, elbow, old_wrist, goal)
        _set_world_rotation(bpy, armature, upper,
                            rotation_between(elbow - shoulder, middle - shoulder) @ original_upper_rotation)
        world = armature.matrix_world
        elbow_now, wrist_now = (np.asarray(world @ bone.head).copy() for bone in (lower, hand))
        lower_rotation = np.asarray((world @ lower.matrix).to_quaternion().to_matrix())
        _set_world_rotation(bpy, armature, lower,
                            rotation_between(wrist_now - elbow_now, reached - elbow_now) @ lower_rotation)
        current_hand = np.asarray((world @ hand.matrix).to_quaternion().to_matrix())
        delta = Matrix((wrist_rotation @ current_hand.T).tolist()).to_quaternion()
        axis = np.asarray(world @ hand.head) - np.asarray(world @ lower.head)
        axis /= np.linalg.norm(axis)
        twist = 2 * np.arctan2(np.dot(np.asarray(delta)[1:], axis), delta.w)
        twist = (twist + np.pi) % (2 * np.pi) - np.pi
        twist = float(np.clip(twist * .75, -np.pi / 2, np.pi / 2))
        rotated = Quaternion(axis, twist) @ (world @ lower.matrix).to_quaternion()
        _set_world_rotation(bpy, armature, lower, np.asarray(rotated.to_matrix()))
        _world_matrix(bpy, armature, hand, wrist_rotation)
        for finger, values in fingers.items():
            for index, segment in enumerate("１２３"):
                bone = bones[prefix + finger + segment]
                angles = references[finger][index].copy()
                angles[0] = np.radians(values[index])
                if index == 0:
                    angles[2] += np.radians(finger_spread[finger])
                bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = Euler(angles, "XYZ").to_quaternion()
        for index, name in enumerate(thumb_names):
            bones[name].rotation_mode = "QUATERNION"
            bones[name].rotation_quaternion = Euler(thumb_reference[index] + np.radians(thumb_delta[index]), "XYZ").to_quaternion()
        bpy.context.view_layer.update()

    def evaluate():
        apply()
        skin = _skin_points(bpy, selections, box_world)
        all_points = _hand_points(skin, side_index, "all")
        distances = box_signed_distance(all_points, size)
        penetration = np.maximum(-distances + .0008, 0)
        score = 40 * np.mean(penetration ** 2) + 8 * float(penetration.max()) ** 2
        limb = _hand_points(skin, side_index, "limb")
        limb_penetration = np.maximum(-box_signed_distance(limb, size) + .001, 0)
        score += 20 * np.mean(limb_penetration ** 2) + 6 * float(limb_penetration.max()) ** 2
        palm = _hand_points(skin, side_index, "palm")
        if len(palm):
            contact = np.sort(np.abs(box_signed_distance(palm, size)))[:max(3, len(palm) // 8)]
            score += 2 * float(np.mean(contact ** 2))
        for finger in (*fingers.keys(), "親指"):
            points = _hand_points(skin, side_index, "tips", finger)
            if len(points):
                gaps = np.sort(np.abs(box_signed_distance(points, size)))[:max(3, len(points) // 5)]
                score += float(np.mean(gaps ** 2)) * (1 if finger == "親指" else 2)
        score += .000001 * sum(float(np.sum(((values - (5, 80, 20)) / 90) ** 2)) for values in fingers.values())
        return float(score)

    try:
        # Coordinate search is deterministic, using real skinned vertices. Distances
        # in metres keep the collision/contact terms independent of pixel framing.
        best = evaluate()
        for step_position, step_angle in ((.012, 25.), (.006, 15.), (.003, 8.), (.0015, 4.)):
            for _ in range(2):
                for axis in (0, 2, 1):
                    original, winner = position[axis], position[axis]
                    for value in (original - step_position, original, original + step_position):
                        if axis == 0 and not size[0] / 2 <= sign * value <= size[0] / 2 + .055:
                            continue
                        if axis == 1 and not -size[1] / 2 - .06 <= value <= -size[1] / 2 - .025:
                            continue
                        position[axis] = value
                        score = evaluate()
                        if score < best:
                            best, winner = score, value
                    position[axis] = winner
                for finger, values in fingers.items():
                    for joint, (low, high) in enumerate(((-15., 65.), (30., 105.), (-5., 55.))):
                        original, winner = values[joint], values[joint]
                        for value in np.unique(np.clip((original - step_angle, original, original + step_angle), low, high)):
                            values[joint] = value
                            score = evaluate()
                            if score < best:
                                best, winner = score, value
                        values[joint] = winner
                    original, winner = finger_spread[finger], finger_spread[finger]
                    for value in np.unique(np.clip((original - step_angle, original, original + step_angle), -15, 15)):
                        finger_spread[finger] = value
                        score = evaluate()
                        if score < best:
                            best, winner = score, value
                    finger_spread[finger] = winner
                # Thumb opposition is measured through its skin, not a guessed fixed angle.
                for joint in range(len(thumb_names)):
                    for axis in (0, 1, 2):
                        original, winner = thumb_delta[joint, axis], thumb_delta[joint, axis]
                        for value in np.unique(np.clip((original - step_angle, original, original + step_angle), -65, 65)):
                            thumb_delta[joint, axis] = value
                            score = evaluate()
                            if score < best:
                                best, winner = score, value
                        thumb_delta[joint, axis] = winner
        apply()
        # A corner needs a combined outward/upward separation. Solving either
        # axis alone can move the wrist centimetres to fix a 2 mm fingertip overlap.
        for _ in range(3):
            skin = _skin_points(bpy, selections, box_world)
            points = _hand_points(skin, side_index, "all")
            for selection, coords in skin:
                triangles = coords[selection["hands"][side_index]["triangles"]]
                hits = box_triangle_intersections(triangles, size, inset=.0002)
                for triangle in triangles[hits]:
                    extra = [triangle[0] + u / 8 * (triangle[1] - triangle[0]) + v / 8 * (triangle[2] - triangle[0])
                             for u in range(9) for v in range(9 - u)]
                    points = np.concatenate((points, extra))
            margin = .002
            if np.min(box_signed_distance(points, size)) >= margin:
                break
            best_shift, best_length = None, float("inf")
            for outward in np.arange(0, .0401, .001):
                for upward in np.arange(0, .0301, .001):
                    shifted = points + (sign * outward, 0, upward)
                    needs_back = (np.abs(shifted[:, 0]) < size[0] / 2 + margin) & (np.abs(shifted[:, 2]) < size[2] / 2 + margin)
                    backward = max(0., float(np.max(shifted[needs_back, 1], initial=-np.inf)) + size[1] / 2 + margin)
                    if backward > .04:
                        continue
                    length = outward * outward + upward * upward + backward * backward
                    if length >= best_length:
                        continue
                    shift = np.array((sign * outward, -backward, upward))
                    if np.min(box_signed_distance(points + shift, size)) >= margin - 1e-9:
                        best_shift, best_length = shift, length
            if best_shift is None:
                break
            position += best_shift
            apply()
        quaternions = {name: list(bones[name].rotation_quaternion) for name in names}
        skin = _skin_points(bpy, selections, box_world)
        distances = box_signed_distance(_hand_points(skin, side_index, "all"), size)
        return {"position": position, "rotation": local_rotation, "finger_quaternions": quaternions,
                "sign": sign, "skin_distance_min_mm": float(distances.min() * 1000),
                "score": best, "finger_degrees": {name: values.tolist() for name, values in fingers.items()},
                "finger_spread_degrees": finger_spread.copy(),
                "thumb_delta_degrees": thumb_delta.tolist()}
    finally:
        for name, basis in saved.items():
            bones[name].matrix_basis = basis
        bpy.context.view_layer.update()


def _clear_body(bpy: Any, selections: list[dict], box: Any, size: np.ndarray,
                frame_start: int, frames: int) -> dict:
    """Keep the rigid box in front of head/torso surfaces, including attached hair."""
    from mathutils import Vector

    matrices, required = [], []
    for frame in range(frame_start, frame_start + frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrix = np.asarray(box.matrix_world).copy()
        matrices.append(matrix)
        skin = _skin_points(bpy, selections, matrix)
        body = np.concatenate([coords[selection["body"]] for selection, coords in skin if len(selection["body"])])
        corridor = (np.abs(body[:, 0]) < size[0] / 2 + .006) & (np.abs(body[:, 2]) < size[2] / 2 + .006)
        corridor &= body[:, 1] < size[1] / 2
        required.append(max(0., float(np.max(body[corridor, 1], initial=-np.inf)) + size[1] / 2 + .006))
    required = np.asarray(required)
    # A max filter followed by a smaller smoothing window remains above every
    # required displacement, without introducing a sharp one-frame collision fix.
    extended = np.pad(required, (3, 3), mode="edge")
    envelope = np.max(np.lib.stride_tricks.sliding_window_view(extended, 7), axis=1)
    smooth = np.convolve(np.pad(envelope, (2, 2), mode="edge"), np.array([1, 2, 3, 2, 1]) / 9, mode="valid")
    for index, frame in enumerate(range(frame_start, frame_start + frames)):
        matrix = matrices[index]
        box.location = Vector(matrix[:3, 3] + matrix[:3, 1] * smooth[index])
        box.keyframe_insert("location", frame=frame)
    return {"forward_shift_max_mm": float(smooth.max() * 1000),
            "forward_shift_per_frame_mm": (smooth * 1000).tolist(),
            "reason": "6 mm clearance from evaluated head/torso-weighted skin; fixed box size and unchanged root"}


def apply_box_hand_contact(bpy: Any, armature: Any, box: Any, *, frame_start: int, frames: int,
                           size: tuple[float, float, float], mesh_objects: list[Any] | None = None) -> dict:
    """Bake two local arm chains and skin-fitted hand poses against a rigid box."""
    from mathutils import Matrix, Quaternion

    from motionviewer.core.limb_ik import solve_two_bone_positions

    from .mmd_contact import _set_world_rotation
    from .retarget._animate import _linearize
    from .retarget.mmd_solve import rotation_between

    size = np.asarray(size, float)
    meshes = mesh_objects if mesh_objects is not None else [
        obj for obj in bpy.data.objects if obj.type == "MESH"
        and any(modifier.type == "ARMATURE" and modifier.object == armature for modifier in obj.modifiers)]
    selections = _mesh_selections(armature, meshes)
    if any(sum(len(item["hands"][side]["all"]) for item in selections) < 30 for side in range(2)):
        raise ValueError("Box contact needs reliable hand skin selections")
    scene = bpy.context.scene
    body_clearance = _clear_body(bpy, selections, box, size, frame_start, frames)
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    templates = [_calibrate_hand(bpy, armature, selections, box, size, prefix, side)
                 for side, prefix in enumerate(("左", "右"))]
    from motionviewer.core.prop_fit import plan_box_elbows

    original_points = np.empty((frames, 2, 3, 3))
    box_matrices = np.empty((frames, 4, 4))
    for offset in range(frames):
        scene.frame_set(frame_start + offset)
        bpy.context.view_layer.update()
        box_matrices[offset] = np.asarray(box.matrix_world)
        for side, prefix in enumerate(("左", "右")):
            original_points[offset, side] = [np.asarray(armature.matrix_world @ armature.pose.bones[prefix + name].head)
                                             for name in ("腕", "ひじ", "手首")]
    elbow_plans = []
    for side in range(2):
        goals = np.einsum("nij,j->ni", box_matrices[:, :3, :3], templates[side]["position"]) + box_matrices[:, :3, 3]
        elbow_plans.append(plan_box_elbows(original_points[:, side, 0], original_points[:, side, 1],
                                          original_points[:, side, 2], goals, box_matrices[:, :3, 3],
                                          box_matrices[:, :3, :3], size))
    source_roots = []
    rows = []
    previous_twist = [None, None]
    for frame in range(frame_start, frame_start + frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        source_roots.append(np.asarray(armature.matrix_world).copy())
        box_world = np.asarray(box.matrix_world).copy()
        row = {"frame": frame, "hands": []}
        for side, prefix in enumerate(("左", "右")):
            template = templates[side]
            upper, lower, hand = (armature.pose.bones[prefix + name] for name in ("腕", "ひじ", "手首"))
            world = armature.matrix_world
            shoulder, elbow, wrist = (np.asarray(world @ bone.head).copy() for bone in (upper, lower, hand))
            original_rotations = [(world @ bone.matrix).to_quaternion().to_matrix() for bone in (upper, lower, hand)]
            original_local = [bone.rotation_quaternion.copy() for bone in (upper, lower, hand)]
            goal = box_world[:3, 3] + box_world[:3, :3] @ template["position"]
            middle, reached, reach_error = solve_two_bone_positions(shoulder, elbow, wrist, goal)
            middle = elbow_plans[side][0][frame - frame_start]
            _set_world_rotation(bpy, armature, upper,
                                rotation_between(elbow - shoulder, middle - shoulder) @ np.asarray(original_rotations[0]))
            elbow_now, wrist_now = (np.asarray(world @ bone.head).copy() for bone in (lower, hand))
            lower_rotation = np.asarray((world @ lower.matrix).to_quaternion().to_matrix()).copy()
            _set_world_rotation(bpy, armature, lower,
                                rotation_between(wrist_now - elbow_now, reached - elbow_now) @ lower_rotation)
            desired_hand = box_world[:3, :3] @ template["rotation"]
            # Share pronation with the forearm instead of placing the entire palm
            # reorientation in the wrist. Rotation about the forearm axis keeps
            # the solved wrist position unchanged.
            current_hand = np.asarray((world @ hand.matrix).to_quaternion().to_matrix())
            delta = Matrix((desired_hand @ current_hand.T).tolist()).to_quaternion()
            axis = np.asarray(world @ hand.head) - np.asarray(world @ lower.head)
            axis /= np.linalg.norm(axis)
            twist = 2 * np.arctan2(np.dot(np.asarray(delta)[1:], axis), delta.w)
            twist = (twist + np.pi) % (2 * np.pi) - np.pi
            if previous_twist[side] is not None:
                twist += 2 * np.pi * round((previous_twist[side] - twist) / (2 * np.pi))
            previous_twist[side] = twist
            twist = float(np.clip(twist * .75, -np.pi / 2, np.pi / 2))
            forearm_rotation = (world @ lower.matrix).to_quaternion()
            rotated = Quaternion(axis, twist) @ forearm_rotation
            _set_world_rotation(bpy, armature, lower, np.asarray(rotated.to_matrix()))
            _set_world_rotation(bpy, armature, hand, desired_hand)
            for name, quaternion in template["finger_quaternions"].items():
                armature.pose.bones[name].rotation_mode = "QUATERNION"
                armature.pose.bones[name].rotation_quaternion = Quaternion(quaternion)
                armature.pose.bones[name].keyframe_insert("rotation_quaternion", frame=frame, group=name)
            bpy.context.view_layer.update()
            for bone in (upper, lower, hand):
                bone.keyframe_insert("rotation_quaternion", frame=frame, group=bone.name)
            after = [(world @ bone.matrix).to_quaternion().to_matrix() for bone in (upper, lower, hand)]
            row["hands"].append({"side": prefix, "wrist_displacement_mm": float(np.linalg.norm(goal - wrist) * 1000),
                                 "reach_error_mm": reach_error * 1000,
                                 "forearm_twist_degrees": float(np.degrees(twist)),
                                 "elbow_pole_change_degrees": float(elbow_plans[side][1][frame - frame_start]),
                                 "local_rotation_change_degrees": [min(float(np.degrees(a.rotation_difference(bone.rotation_quaternion).angle)),
                                                                         360 - float(np.degrees(a.rotation_difference(bone.rotation_quaternion).angle)))
                                                                     for a, bone in zip(original_local, (upper, lower, hand), strict=True)],
                                 "actual_wrist_error_mm": float(np.linalg.norm(np.asarray(world @ hand.head) - goal) * 1000),
                                 "world_rotation_change_degrees": [_degrees_between(a, b) for a, b in zip(original_rotations, after, strict=True)]})
        rows.append(row)
    _linearize(armature)
    body_worst = 0.
    for row, source_root in zip(rows, source_roots, strict=True):
        scene.frame_set(row["frame"])
        bpy.context.view_layer.update()
        skin = _skin_points(bpy, selections, np.asarray(box.matrix_world))
        row["root_matrix_max_change"] = float(np.max(np.abs(source_root - np.asarray(armature.matrix_world))))
        for side, hand_report in enumerate(row["hands"]):
            distances = box_signed_distance(_hand_points(skin, side, "all"), size)
            hand_report["skin_penetration_max_mm"] = max(0., -float(distances.min()) * 1000)
            hand_report["skin_contact_gap_mm"] = float(np.min(np.abs(distances)) * 1000)
            hand_report["fingertip_contact_mm"] = {}
            triangles = np.concatenate([coords[selection["hands"][side]["triangles"]] for selection, coords in skin])
            hand_report["skin_triangle_box_intersections_0_2mm"] = int(box_triangle_intersections(triangles, size, inset=.0002).sum())
            for finger in ("人指", "中指", "薬指", "小指", "親指"):
                points = _hand_points(skin, side, "tips", finger)
                if len(points):
                    hand_report["fingertip_contact_mm"][finger] = float(np.min(np.abs(box_signed_distance(points, size))) * 1000)
        body = [coords[selection["body"]] for selection, coords in skin if len(selection["body"])]
        body_distance = box_signed_distance(np.concatenate(body), size) if body else np.array([float("inf")])
        row["body_skin_penetration_max_mm"] = max(0., -float(body_distance.min()) * 1000)
        body_triangles = np.concatenate([coords[selection["body_triangles"]] for selection, coords in skin])
        row["body_triangle_box_intersections_0_2mm"] = int(box_triangle_intersections(body_triangles, size, inset=.0002).sum())
        all_triangles = np.concatenate([coords[selection["triangles"]] for selection, coords in skin])
        near = np.all(all_triangles.min(axis=1) < size / 2, axis=1) & np.all(all_triangles.max(axis=1) > -size / 2, axis=1)
        row["actor_triangle_box_intersections_0_2mm"] = int(box_triangle_intersections(all_triangles[near], size, inset=.0002).sum())
        body_worst = max(body_worst, row["body_skin_penetration_max_mm"])
    return {"enabled": True, "evaluation_stage": "pre_physics",
            "method": "fixed rigid box; actual hand skin calibration; two-bone arm IK; wrist/finger rotations",
            "body_clearance": body_clearance,
            "source_files_modified": False, "root_matrix_max_change": max(row["root_matrix_max_change"] for row in rows),
            "hand_skin_penetration_max_mm": max(hand["skin_penetration_max_mm"] for row in rows for hand in row["hands"]),
            "body_skin_penetration_max_mm": body_worst,
            "wrist_displacement_max_mm": max(hand["wrist_displacement_mm"] for row in rows for hand in row["hands"]),
            "actual_wrist_error_max_mm": max(hand["actual_wrist_error_mm"] for row in rows for hand in row["hands"]),
            "elbow_pole_max_degrees": max(float(np.abs(plan[1]).max()) for plan in elbow_plans),
            "elbow_pole_max_step_degrees": max(float(np.max(np.abs(np.diff(plan[1])), initial=0)) for plan in elbow_plans),
            "actor_triangle_box_intersections_max": max(row["actor_triangle_box_intersections_0_2mm"] for row in rows),
            "body_triangle_box_intersections_max": max(row["body_triangle_box_intersections_0_2mm"] for row in rows),
            "hand_triangle_box_intersections_max": max(hand["skin_triangle_box_intersections_0_2mm"] for row in rows for hand in row["hands"]),
            "templates": [{key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in template.items()} for template in templates], "frames": rows,
            "limitations": "Evaluated skin vertices/edge/face samples and per-animation-frame triangle/box intersection tests "
                           "(0.2 mm box inset); no continuous collision test between frames. Whole-actor triangle residuals "
                           "include forearm cuffs, bracelets, hair and clothing and must not be hidden by hand-only metrics."}
