from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from RunQiandaoFullFrameCombNavigation import iter_sensor_schedule
from QiandaoVirtualNavigation import deterministic_schedule


@pytest.mark.parametrize('duration,period', [(1., .33), (10.1, .3299999237060547), (313., .2)])
def test_streaming_events_preserve_pilot_exact_times_priorities_and_counts(duration, period):
    assert list(iter_sensor_schedule(duration, period)) == deterministic_schedule(duration, period)


def test_invalid_schedule_rejected_and_generator_is_streaming():
    iterator = iter_sensor_schedule(50000., .33)
    assert iter(iterator) is iterator
    assert next(iterator) == (0., 'sonar')
    assert next(iterator) == (.02, 'imu')
    with pytest.raises(ValueError):
        list(iter_sensor_schedule(10, 0))
