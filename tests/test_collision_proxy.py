import numpy as np
import pytest

from motionviewer.core.collision_proxy import surface_cells


def test_long_triangle_is_clipped_and_surface_area_is_preserved():
    vertices = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cells = surface_cells(vertices, np.array([[0, 1, 2]]), 0.2)
    assert len(cells) > 10
    # Convex polygon area in YZ, sorted around its centroid.
    area = 0.
    for points in cells:
        assert np.max(np.ptp(points[:, 1:], axis=0)) <= .20000001
        yz = points[:, 1:]
        angles = np.arctan2(*(yz - yz.mean(axis=0))[:, ::-1].T)
        yz = yz[np.argsort(angles)]
        area += abs(np.dot(yz[:, 0], np.roll(yz[:, 1], 1)) - np.dot(yz[:, 1], np.roll(yz[:, 0], 1))) / 2
    assert area == pytest.approx(.5, abs=1e-8)
    assert all(np.linalg.matrix_rank(p - p.mean(axis=0), tol=1e-7) >= 2 for p in cells)


def test_zero_area_triangles_cannot_become_zero_volume_bullet_hulls():
    # Three distinct collinear points survive unique(), so a point-count-only
    # guard would send these cells to Bullet and one-axis thickening would fail.
    vertices = np.array([[0., 0., 0.], [0., .24, .24], [0., .48, .48]])
    assert surface_cells(vertices, np.array([[0, 1, 2]])) == []


def test_disconnected_blades_do_not_fill_large_opening():
    vertices = np.array([[0, 0, 0], [0, .1, 0], [.01, 0, .1],
                         [0, 1, 0], [0, 1.1, 0], [.01, 1, .1]])
    cells = surface_cells(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    assert all(p[:, 1].max() <= .1 or p[:, 1].min() >= 1 for p in cells)


@pytest.mark.parametrize("size", [0, -1, float("nan")])
def test_invalid_cell_size(size):
    with pytest.raises(ValueError):
        surface_cells(np.zeros((3, 3)), np.array([[0, 1, 2]]), size)
