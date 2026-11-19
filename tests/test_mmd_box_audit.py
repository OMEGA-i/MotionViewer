from types import SimpleNamespace

import numpy as np
import pytest

from motionviewer.blender.mmd_box_contact import audit_animated_box_clearance


@pytest.mark.parametrize("fail", [False, True])
def test_box_audit_reads_evaluated_frames_and_restores_frame(fail):
    scene = SimpleNamespace(frame_current=99)
    scene.frame_set = lambda frame: setattr(scene, "frame_current", frame)
    context = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(update=lambda: None),
                              evaluated_depsgraph_get=lambda: object())

    class Vertices:
        def __len__(self):
            return 3

        def foreach_get(self, name, destination):
            if fail:
                raise RuntimeError("evaluation failed")
            points = np.array([[-2, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
            if scene.frame_current == 1:
                points[:, 2] = 2
            destination[:] = points.ravel()

    data = SimpleNamespace(vertices=Vertices(), calc_loop_triangles=lambda: None,
                           loop_triangles=[SimpleNamespace(vertices=(0, 1, 2))])
    cleared = []
    obj = SimpleNamespace(matrix_world=np.eye(4), to_mesh=lambda: data,
                          to_mesh_clear=lambda: cleared.append(scene.frame_current))
    mesh = SimpleNamespace(evaluated_get=lambda depsgraph: obj)
    kwargs = dict(size=(1, 1, 1), frame_start=1, frames=2, evaluation_stage="post_baked_bullet")
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            audit_animated_box_clearance(SimpleNamespace(context=context),
                                        SimpleNamespace(matrix_world=np.eye(4)), [mesh], **kwargs)
        assert cleared == [1]
    else:
        report = audit_animated_box_clearance(SimpleNamespace(context=context),
                                             SimpleNamespace(matrix_world=np.eye(4)), [mesh], **kwargs)
        assert report["evaluation_stage"] == "post_baked_bullet"
        assert report["frames_with_intersections"] == [2]
        assert report["actor_triangle_intersections_max"] == 1
        assert report["sampled_penetration_max_mm"] == pytest.approx(500)
        assert cleared == [1, 2]
    assert scene.frame_current == 99
