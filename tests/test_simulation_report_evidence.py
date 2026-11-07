from pathlib import Path
import csv
import gzip
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from BuildSeabedSimulationReportEvidence import summarize, segmented, write_csv, assert_metrics


def test_actual_full_error_metrics_and_invalid_inputs():
    result = summarize([0., 3., 4.])
    assert result['rmse_m'] == pytest.approx(np.sqrt(25/3))
    assert result['final_m'] == result['max_m'] == 4.
    for value in ([], [np.nan], [[1.]]):
        with pytest.raises(ValueError): summarize(value)
    with pytest.raises(AssertionError): assert_metrics([0., 3., 4.], {'rmse_m': 0.})


def test_segment_statistics_include_boundary_samples_and_are_not_resampled():
    rows = segmented(np.array([0., 99., 100., 201.]), {'global': np.array([0., 4., 3., 2.])})
    assert sum(row[2] for row in rows) == 4
    assert [row[0] for row in rows] == [0., 100., 200.]
    assert rows[0][4] == pytest.approx(np.sqrt(8.))
    with pytest.raises(ValueError): segmented(np.array([2., 1.]), {'global': [1., 2.]})


def test_lossless_csv_and_gzip_roundtrip_preserve_all_native_float_samples(tmp_path):
    data = [[.1, 1.2345678901234567], [100., -1e-15]]
    for name in ('full.csv', 'full.csv.gz'):
        path = tmp_path/name
        write_csv(path, ['t', 'x'], data)
        opener = gzip.open if name.endswith('.gz') else Path.open
        with opener(path, mode='rt' if name.endswith('.gz') else 'r', newline='') as stream:
            reader = csv.reader(stream); assert next(reader) == ['t', 'x']
            actual = [[float(x) for x in row] for row in reader]
        np.testing.assert_array_equal(actual, data)
        with pytest.raises(FileExistsError): write_csv(path, ['t', 'x'], data)
