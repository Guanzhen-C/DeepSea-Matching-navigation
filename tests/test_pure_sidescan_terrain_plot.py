from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from PlotPureSideScanTerrain import terrain_mesh, select_three_trajectories, require_full_plot_scope


def test_display_mesh_uses_calibrated_centered_coordinates_retains_boundaries_and_unknown_cells():
    z = np.arange(35.).reshape(5, 7); valid = np.ones_like(z, bool); valid[2, 2] = False
    x, y, shown = terrain_mesh(z, valid, .3, step=2)
    np.testing.assert_allclose(x[0], [-.9, -.3, .3, .9], atol=1e-15)
    np.testing.assert_allclose(y[:, 0], [-.6, 0., .6], atol=1e-15)
    assert np.isnan(shown[1, 1]) and shown[-1, -1] == z[-1, -1]
    assert z[2, 2] == 16.  # original terrain is not overwritten or filled


def test_exactly_three_native_trajectories_use_new_global_without_offsets_or_decimation():
    p = np.arange(120.).reshape(10, 4, 3)
    result = select_three_trajectories(p)
    assert list(result) == ['Truth', 'Local', 'Global']
    for name, k in [('Truth', 0), ('Local', 2), ('Global', 3)]:
        np.testing.assert_array_equal(result[name], p[:, k])
        assert len(result[name]) == 10
    result['Global'][0] = 0
    assert p[0, 3, 0] == 9.
    p[0, 3, 0] = np.nan
    with pytest.raises(ValueError): select_three_trajectories(p)


def test_short_prefix_cannot_be_labeled_as_the_full_terrain_plot():
    require_full_plot_scope({'full_closed_loop_requested': True}, {'complete_planned_route_verified': True})
    for original, review in [({}, {}), ({'full_closed_loop_requested': False}, {'complete_planned_route_verified': True}),
                             ({'full_closed_loop_requested': True}, {})]:
        with pytest.raises(ValueError, match='prefix'): require_full_plot_scope(original, review)
