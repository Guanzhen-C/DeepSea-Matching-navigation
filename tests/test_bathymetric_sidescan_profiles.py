from __future__ import annotations

import json
import math

import pytest

from oceansim.sensors.bathymetric_sidescan_profiles import (
    HIGH_V2_PROFILE,
    STANDARD_V2_PROFILE,
    acoustic_two_way_travel_time_s,
    get_v2_profile,
    maximum_supported_ping_rate_hz,
    required_distinguishable_pings_in_water,
)


@pytest.mark.parametrize(
    (
        "profile",
        "mode",
        "sample_spacing_m",
        "coverage_m",
        "breakpoint_m",
    ),
    (
        (STANDARD_V2_PROFILE, "standard", 0.15, 1.2, 60.0),
        (HIGH_V2_PROFILE, "high", 0.075, 0.6, 30.0),
    ),
)
def test_v2_profiles_preserve_documented_mode_parameters(
    profile,
    mode: str,
    sample_spacing_m: float,
    coverage_m: float,
    breakpoint_m: float,
) -> None:
    assert get_v2_profile(mode) is profile
    assert profile.rows_per_side_per_ping == 8
    assert profile.along_track_sample_spacing_m == sample_spacing_m
    assert profile.along_track_coverage_per_ping_m == coverage_m
    assert profile.constant_resolution_breakpoint_m == breakpoint_m
    assert profile.near_range_along_track_resolution_m == sample_spacing_m
    assert profile.horizontal_beam_opening_deg == 0.14
    assert profile.range_sample_spacing_m == 0.02
    assert profile.range_resolution_m == 0.025
    assert profile.maximum_range_m == 250.0
    assert profile.range_bins_per_side == 12_500
    assert profile.range_bins_per_side == round(
        profile.maximum_range_m / profile.range_sample_spacing_m
    )


@pytest.mark.parametrize("profile", (STANDARD_V2_PROFILE, HIGH_V2_PROFILE))
def test_far_range_resolution_uses_degrees(profile) -> None:
    assert profile.along_track_resolution_m(
        profile.constant_resolution_breakpoint_m
    ) == profile.near_range_along_track_resolution_m
    expected_at_150_m = math.sin(math.radians(0.14)) * 150.0
    assert profile.along_track_resolution_m(150.0) == pytest.approx(
        expected_at_150_m
    )
    assert expected_at_150_m == pytest.approx(0.3665, abs=1.0e-4)
    assert profile.along_track_resolution_m(150.0) != pytest.approx(
        math.sin(0.14) * 150.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        profile.along_track_resolution_m(-0.1)
    with pytest.raises(ValueError, match="exceeds"):
        profile.along_track_resolution_m(250.1)


def test_v2_metadata_is_json_serializable_and_marks_unknowns() -> None:
    metadata = STANDARD_V2_PROFILE.to_metadata()
    decoded = json.loads(json.dumps(metadata, allow_nan=False))

    assert decoded["mode"] == "standard"
    assert decoded["source_document_sha256"] == (
        "d078dbfae5776192a3f84d186c9a22b29859e53ee1502a187ab6ba4170be4866"
    )
    assert decoded["source_device_identified"] is False
    assert decoded["manufacturer"] is None
    assert decoded["model"] is None
    expected_unknowns = {
        "manufacturer",
        "model",
        "acoustic_frequency_hz",
        "ping_rate_hz",
        "vertical_beam_opening_deg",
        "pulse_duration_s",
        "receiver_bandwidth_hz",
        "source_level_db_re_1upa_at_1m",
        "bathymetry_accuracy_m",
        "intensity_unit",
        "vehicle_speed_m_s",
        "operating_altitude_m",
        "line_spacing_m",
        "full_swath_width_m",
    }
    assert set(decoded["unprovided_fields"]) == expected_unknowns
    assert "radians" in decoded["far_range_resolution_equation"]


def test_unknown_v2_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown V2 profile"):
        get_v2_profile("ultra")


def test_full_range_schedule_requires_explicit_distinguishable_pings() -> None:
    travel_time = acoustic_two_way_travel_time_s(250.0, 1500.0)

    assert travel_time == pytest.approx(1.0 / 3.0)
    assert required_distinguishable_pings_in_water(3.0, 250.0, 1500.0) == 1
    assert required_distinguishable_pings_in_water(5.0, 250.0, 1500.0) == 2
    assert required_distinguishable_pings_in_water(10.0, 250.0, 1500.0) == 4
    assert maximum_supported_ping_rate_hz(250.0, 1500.0, 2) == pytest.approx(
        6.0
    )
    assert maximum_supported_ping_rate_hz(250.0, 1500.0, 4) == pytest.approx(
        12.0
    )
