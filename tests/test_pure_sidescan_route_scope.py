from pathlib import Path
import sys
import json
from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from RunPureSideScanFeatureShort import resolve_route_distance, route_scope_metadata


def test_explicit_route_prefix_can_include_turns_without_being_called_full():
    assert resolve_route_distance(2000., False, 16488.) == 2000.
    assert resolve_route_distance(None, False, 16488.) == 400.
    assert resolve_route_distance(None, True, 16488.) == 16488.


@pytest.mark.parametrize('distance,full', [(0., False), (-1., False), (float('nan'), False),
                                         (17000., False), (400., True)])
def test_route_request_cannot_silently_loop_or_change_scope(distance, full):
    with pytest.raises(ValueError): resolve_route_distance(distance, full, 16488.)


def test_real_numpy_route_arrays_produce_strict_JSON_native_scope_metadata():
    route = SimpleNamespace(length_m=np.float64(16488.), ends_m=np.array([1300., 1341.]), speed_mps=.8)
    metadata = route_scope_metadata(2000., False, route)
    assert metadata['contains_programmed_turns'] is True
    assert metadata['full_closed_loop_requested'] is False
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
