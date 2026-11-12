from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from RandomTriggeredSonarNoise import apply_scenario, counter_complex, draw_scenario, scenario_sha, colour_kernel
from TriggeredSonarNoise import build_source_model


def profile():
    wave = np.broadcast_to(np.cos(2 * np.pi * np.arange(48) / 48), (8, 48))
    tiles = [dict(side=i, candidate_peak_to_background_power=100.,
        folded_log_amplitude_templates=wave.tolist(),
        candidate_complex_range_PSD=np.full(512, 1 / 512).tolist(),
        beam_covariance_real=np.eye(8).tolist(), beam_covariance_imag=np.zeros((8, 8)).tolist(),
        candidate_RMS_over_FIR_anchor_RMS=.5) for i in (0, 1)]
    fitted = dict(fitted_record_frequency_q05_q50_q95=[.15, .17, .18],
                  fitted_range_frequency_q05_q50_q95=[.002, .003, .004])
    p = dict(tiles=tiles, curved_phase_geometry={'0': fitted, '1': fitted})
    p['triggered_source_model'] = build_source_model(p, 744, 12189)
    return p


def test_coordinate_gaussian_is_exactly_crop_consistent_and_seeded():
    whole = counter_complex(21, np.arange(100)[:, None], np.arange(-16, 240)[None])
    crop = counter_complex(21, np.arange(20, 90)[:, None], np.arange(16, 180)[None])
    np.testing.assert_array_equal(whole[20:90, 32:196], crop)
    assert not np.array_equal(whole, counter_complex(22, np.arange(100)[:, None], np.arange(-16, 240)[None]))
    assert abs(np.mean(whole)) < .03
    assert np.mean(np.abs(whole)**2) == pytest.approx(1, abs=.03)


def test_scenario_randomizes_count_positions_phases_spacing_strength_and_remains_reproducible():
    p = profile()
    a = draw_scenario(p, (500, 1200), 0, 33)
    assert scenario_sha(a) == scenario_sha(draw_scenario(p, (500, 1200), 0, 33))
    scenarios = [draw_scenario(p, (500, 1200), 0, seed) for seed in range(25)]
    assert len({len(s['events']) for s in scenarios}) > 2
    for key in ('centre_record', 'centre_range', 'spacing', 'phase', 'carrier_phase', 'crest_width_scale'):
        assert len({s['events'][0][key] for s in scenarios}) == len(scenarios)
    assert all(e['centre_range'] < .12 * 1200 or e['centre_range'] > .88 * 1199
               for s in scenarios for e in s['events'])
    assert not a['separate_illumination_field'] and not a['original_scene_warped']


@pytest.mark.parametrize('domain', range(4))
def test_whole_and_crop_use_same_noise_for_every_real_data_type(domain):
    torch.set_num_threads(2)
    p = profile()
    values = torch.ones((100, 320), dtype=torch.complex64 if domain >= 2 else torch.float32) * .1
    values[:, :30] = 0
    gate = np.ones(values.shape, np.float32)
    gate[:, :70] = 0  # nonzero dark water is protected, not painted black
    before = values.clone()
    scenario = draw_scenario(p, values.shape, domain, 47, side=0 if domain == 3 else None)
    kwargs = dict(reference_RMS=.1, beam=2 if domain == 3 else None)
    whole, metadata = apply_scenario(values, p, scenario, support=gate, **kwargs)
    crop, _ = apply_scenario(values[21:78, 40:240], p, scenario, origin=(21, 40),
                            support=gate[21:78, 40:240], **kwargs)
    torch.testing.assert_close(whole[21:78, 40:240], crop, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    torch.testing.assert_close(whole[:, :70], values[:, :70], rtol=0, atol=0)
    assert torch.any(whole[:, 70:] != values[:, 70:])
    assert whole.is_complex() == (domain >= 2)
    assert not metadata['unmeasured_phase_invented'] and not metadata['reference_filtered']
    assert metadata['patch_and_full_use_identical_noise_generator']
    assert torch.isfinite(whole).all()
    assert metadata['actual_changed_values'] == int((whole != values).sum())


def test_identity_and_no_source_have_no_brightness_change():
    p = profile()
    values = torch.full((40, 200), .1)
    scenario = draw_scenario(p, values.shape, 1, 31)
    output, _ = apply_scenario(values, p, scenario, reference_RMS=.1, identity=True)
    assert torch.equal(output, values)
    scenario['events'] = []
    output, _ = apply_scenario(values, p, scenario, reference_RMS=.1)
    assert torch.equal(output, values)


def test_fully_protected_input_is_not_counted_as_actual_augmented_exposure():
    p = profile()
    values = torch.full((40, 200), .1)
    output, metadata = apply_scenario(values, p, draw_scenario(p, values.shape, 1, 31),
        reference_RMS=.1, support=torch.zeros_like(values))
    assert torch.equal(output, values)
    assert not metadata['identity']
    assert metadata['actual_changed_values'] == 0


def test_colouring_normalizes_only_synthetic_noise_energy():
    p = profile()
    kernel = colour_kernel(p['triggered_source_model']['sides']['0'])
    assert np.square(np.abs(kernel)).sum() == pytest.approx(1, abs=2e-7)
    with pytest.raises(ValueError):
        colour_kernel(p['triggered_source_model']['sides']['0'], taps=32)


@pytest.mark.parametrize('gate', [np.full((40, 200), float('nan')), np.full((40, 200), -1), np.ones((2, 2))])
def test_bad_support_is_refused(gate):
    p = profile()
    values = torch.full((40, 200), .1)
    with pytest.raises(ValueError, match='support'):
        apply_scenario(values, p, draw_scenario(p, values.shape, 1, 31), reference_RMS=.1, support=gate)


def test_identity_sampling_does_not_alias_native_source_schedule():
    import random
    from NoFIRSonarData import plan_source
    records = [dict(id=f'n{i}', role='native_train', domain=3) for i in range(48)]
    records += [dict(id=f'p{d}', role='public_train', domain=d) for d in (0, 1, 2)]
    augmented = set()
    identity = set()
    for index in range(6000):
        item = plan_source(records, 'adapt', index)
        seed = 17618000 + index + 2000000
        if random.Random(seed + 713).random() < .2:
            identity.add(item['id'])
        else:
            augmented.add(item['id'])
    assert {r['id'] for r in records} <= augmented
    assert {r['id'] for r in records} <= identity
