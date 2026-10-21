import numpy as np
import pytest

from standalone.terrain_plot_mesh import measured_surface


def test_display_mesh_retains_native_coordinates_and_no_nodata_bridge():
    z = np.arange(81., dtype=np.float64).reshape(9, 9)
    mask = np.ones((9, 9), dtype=bool)
    mask[4, 4] = False
    points, triangles = measured_surface(z, mask, 1., (-4., 4., -4., 4.), stride=2)
    for triangle in points[triangles]:
        xmin, ymin = np.min(triangle[:, :2], axis=0).astype(int) + 4
        xmax, ymax = np.max(triangle[:, :2], axis=0).astype(int) + 4
        assert mask[ymin:ymax+1, xmin:xmax+1].all()
    for x, y, height in points:
        assert height == z[int(y+4), int(x+4)]


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_display_mesh_rejects_invalid_stride(stride):
    with pytest.raises(ValueError):
        measured_surface(np.zeros((5, 5)), np.ones((5, 5), dtype=bool), 1., (-2., 2., -2., 2.), stride=stride)


def test_display_mesh_rejects_no_measured_cells():
    with pytest.raises(ValueError, match="no fully measured"):
        measured_surface(np.zeros((5, 5)), np.zeros((5, 5), dtype=bool), 1., (-2., 2., -2., 2.), stride=2)
