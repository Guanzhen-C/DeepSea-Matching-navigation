"""Document-derived profiles for the V2 bathymetric side-scan sonar.

The source document describes two imaging modes but does not identify a
manufacturer or product model.  These profiles therefore preserve unknown
hardware quantities as ``None`` instead of inventing calibration values.

The document calls the eight along-track samples "beams".  ``rows_per_side``
uses the less ambiguous image-domain name because this value must not be
confused with the across-track fan-beam count of a conventional multibeam
echo sounder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def acoustic_two_way_travel_time_s(
    maximum_range_m: float, sound_speed_m_s: float
) -> float:
    """Return the monostatic two-way travel time at maximum slant range."""

    maximum_range = float(maximum_range_m)
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(maximum_range) or maximum_range <= 0.0:
        raise ValueError("maximum_range_m must be finite and positive")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and positive")
    return 2.0 * maximum_range / sound_speed


def required_distinguishable_pings_in_water(
    ping_rate_hz: float,
    maximum_range_m: float,
    sound_speed_m_s: float,
) -> int:
    """Return concurrent coded/interleaved slots needed for a ping schedule.

    A value above one does not claim that the V2 source document provides a
    multi-ping waveform.  It makes the extra acquisition capability required
    by a requested rate explicit instead of silently violating two-way travel
    time.
    """

    ping_rate = float(ping_rate_hz)
    if not math.isfinite(ping_rate) or ping_rate <= 0.0:
        raise ValueError("ping_rate_hz must be finite and positive")
    occupancy = ping_rate * acoustic_two_way_travel_time_s(
        maximum_range_m, sound_speed_m_s
    )
    # Preserve the exact integer boundary against harmless floating-point
    # roundoff (for example, 3 Hz * 1/3 s needs one slot, not two).
    return max(1, int(math.ceil(occupancy - 1.0e-12)))


def maximum_supported_ping_rate_hz(
    maximum_range_m: float,
    sound_speed_m_s: float,
    distinguishable_pings_in_water: int = 1,
) -> float:
    """Return the ambiguity-free rate for the configured concurrent slots."""

    slots = int(distinguishable_pings_in_water)
    if slots != distinguishable_pings_in_water or slots < 1:
        raise ValueError("distinguishable_pings_in_water must be a positive integer")
    return slots / acoustic_two_way_travel_time_s(
        maximum_range_m, sound_speed_m_s
    )


@dataclass(frozen=True)
class BathymetricSideScanV2Profile:
    """One operating mode transcribed from the supplied V2 document."""

    mode: str
    rows_per_side_per_ping: int
    along_track_sample_spacing_m: float
    along_track_coverage_per_ping_m: float
    constant_resolution_breakpoint_m: float
    near_range_along_track_resolution_m: float
    horizontal_beam_opening_deg: float
    range_sample_spacing_m: float
    range_resolution_m: float
    maximum_range_m: float
    range_bins_per_side: int

    # The source does not provide these hardware or survey quantities.
    manufacturer: str | None = None
    model: str | None = None
    acoustic_frequency_hz: float | None = None
    ping_rate_hz: float | None = None
    vertical_beam_opening_deg: float | None = None
    pulse_duration_s: float | None = None
    receiver_bandwidth_hz: float | None = None
    source_level_db_re_1upa_at_1m: float | None = None
    bathymetry_accuracy_m: float | None = None
    intensity_unit: str | None = None
    vehicle_speed_m_s: float | None = None
    operating_altitude_m: float | None = None
    line_spacing_m: float | None = None
    full_swath_width_m: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "high"}:
            raise ValueError("mode must be 'standard' or 'high'")
        positive_values = {
            "rows_per_side_per_ping": self.rows_per_side_per_ping,
            "along_track_sample_spacing_m": self.along_track_sample_spacing_m,
            "along_track_coverage_per_ping_m": self.along_track_coverage_per_ping_m,
            "constant_resolution_breakpoint_m": (
                self.constant_resolution_breakpoint_m
            ),
            "near_range_along_track_resolution_m": (
                self.near_range_along_track_resolution_m
            ),
            "horizontal_beam_opening_deg": self.horizontal_beam_opening_deg,
            "range_sample_spacing_m": self.range_sample_spacing_m,
            "range_resolution_m": self.range_resolution_m,
            "maximum_range_m": self.maximum_range_m,
            "range_bins_per_side": self.range_bins_per_side,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rows_per_side_per_ping != int(self.rows_per_side_per_ping):
            raise ValueError("rows_per_side_per_ping must be an integer")
        if self.range_bins_per_side != int(self.range_bins_per_side):
            raise ValueError("range_bins_per_side must be an integer")
        expected_coverage = (
            self.rows_per_side_per_ping * self.along_track_sample_spacing_m
        )
        if not math.isclose(
            self.along_track_coverage_per_ping_m,
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "along-track coverage must equal rows per side times sample spacing"
            )
        expected_bins = self.maximum_range_m / self.range_sample_spacing_m
        if not math.isclose(
            float(self.range_bins_per_side),
            expected_bins,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "range bins per side must equal maximum range divided by spacing"
            )

    def along_track_resolution_m(self, range_m: float) -> float:
        """Return the document's piecewise along-track resolution.

        The document denotes the independent variable by ``R`` and later uses
        it as slant range.  Its exact operational interpretation is retained
        as a source ambiguity.  The far-range equation is evaluated in
        degrees, as required by the stated 0.14 degree opening angle and the
        document's 150 m numerical example.
        """

        distance = float(range_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("range_m must be finite and non-negative")
        if distance > self.maximum_range_m:
            raise ValueError(
                f"range_m exceeds the documented {self.maximum_range_m:g} m limit"
            )
        if distance <= self.constant_resolution_breakpoint_m:
            return self.near_range_along_track_resolution_m
        return math.sin(math.radians(self.horizontal_beam_opening_deg)) * distance

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible values with provenance and ambiguities."""

        metadata: dict[str, Any] = asdict(self)
        metadata.update(
            {
                "schema": "oceansim.bathymetric_sidescan.v2.profile/1",
                "source_document": "多波束测深侧扫作业及成像V2.docx",
                "source_document_sha256": (
                    "d078dbfae5776192a3f84d186c9a22b29859e53ee1502a187ab6ba4170be4866"
                ),
                "source_device_identified": False,
                "far_range_resolution_equation": (
                    "sin(radians(horizontal_beam_opening_deg)) * range_m"
                ),
                "source_ambiguities": [
                    "R is not consistently distinguished as slant or ground range",
                    "250 m is not explicitly identified as per-side or full-swath width",
                    "eight along-track beams are represented here as image rows",
                    "2 cm sampling is not explicitly identified before or after slant correction",
                ],
            }
        )
        metadata["unprovided_fields"] = sorted(
            key for key, value in asdict(self).items() if value is None
        )
        return metadata


STANDARD_V2_PROFILE = BathymetricSideScanV2Profile(
    mode="standard",
    rows_per_side_per_ping=8,
    along_track_sample_spacing_m=0.15,
    along_track_coverage_per_ping_m=1.2,
    constant_resolution_breakpoint_m=60.0,
    near_range_along_track_resolution_m=0.15,
    horizontal_beam_opening_deg=0.14,
    range_sample_spacing_m=0.02,
    range_resolution_m=0.025,
    maximum_range_m=250.0,
    range_bins_per_side=12_500,
)

HIGH_V2_PROFILE = BathymetricSideScanV2Profile(
    mode="high",
    rows_per_side_per_ping=8,
    along_track_sample_spacing_m=0.075,
    along_track_coverage_per_ping_m=0.6,
    constant_resolution_breakpoint_m=30.0,
    near_range_along_track_resolution_m=0.075,
    horizontal_beam_opening_deg=0.14,
    range_sample_spacing_m=0.02,
    range_resolution_m=0.025,
    maximum_range_m=250.0,
    range_bins_per_side=12_500,
)


_V2_PROFILES = {
    STANDARD_V2_PROFILE.mode: STANDARD_V2_PROFILE,
    HIGH_V2_PROFILE.mode: HIGH_V2_PROFILE,
}


def get_v2_profile(mode: str) -> BathymetricSideScanV2Profile:
    """Return a named V2 profile without importing Isaac Sim."""

    key = str(mode).strip().lower()
    try:
        return _V2_PROFILES[key]
    except KeyError as error:
        choices = ", ".join(sorted(_V2_PROFILES))
        raise ValueError(f"unknown V2 profile {mode!r}; choose {choices}") from error


__all__ = [
    "acoustic_two_way_travel_time_s",
    "BathymetricSideScanV2Profile",
    "HIGH_V2_PROFILE",
    "maximum_supported_ping_rate_hz",
    "required_distinguishable_pings_in_water",
    "STANDARD_V2_PROFILE",
    "get_v2_profile",
]
