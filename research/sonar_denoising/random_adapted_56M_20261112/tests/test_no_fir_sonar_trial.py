from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from NoFIRSonarWavelet import NoFIRSonarMWCNN, raw_features, infer_raw, native_context
from NoFIRSonarData import Sources, augment, plan_source, json_text, save


def profile():
    return dict(reference_period_records=[9., 6.], reference_period_mosaic_rows=[36., 24.],
                reference_abs_native_slopes=[.1, .07])


def test_unmeasured_phase_is_not_invented_or_accepted_as_public_envelope():
    a = np.ones((12, 40), np.float32)
    features, q, mask = raw_features(a, 0)
    assert not features[1:].any() and not q.any() and mask.all()
    with pytest.raises(ValueError, match='actual complex phase'):
        raw_features(a.astype(np.complex64), 0)
    with pytest.raises(ValueError, match='actual complex phase'):
        raw_features(a, 3)
    model = NoFIRSonarMWCNN(2)
    with pytest.raises(ValueError, match='unmeasured public phase'):
        model.feature_residual(torch.from_numpy(features[None]), torch.from_numpy(mask[None, None]), domain=0)


def test_actual_public_complex_phase_has_no_invented_sampling_frequency():
    x = np.arange(48)
    z = np.broadcast_to(np.exp(.37j*x), (12, 48)).astype(np.complex64)
    features, _, _ = raw_features(z, 2)
    np.testing.assert_allclose(features[1, :, 1:], np.cos(.37), atol=2e-7)
    np.testing.assert_allclose(features[2, :, 1:], np.sin(.37), atol=2e-7)
    np.testing.assert_array_equal(features[1:, :, 0], 0)


def test_identity_raw_inference_cannot_call_legacy_FIR(monkeypatch):
    import MatchedSonarSignal
    def forbidden(*args, **kwargs):
        raise AssertionError('FIR forbidden in this candidate')
    monkeypatch.setattr(MatchedSonarSignal, 'negative_frontend', forbidden)
    torch.set_num_threads(2)
    rng = np.random.default_rng(31)
    z = (rng.normal(size=(25, 131))+1j*rng.normal(size=(25, 131))).astype(np.complex64)
    z[:, :3] = 0
    model = NoFIRSonarMWCNN(2)
    delta = infer_raw(model, z, 'cpu')
    assert delta.shape == z.shape and not delta.any()
    np.testing.assert_array_equal(z*np.exp(delta), z)


def test_model_training_reaches_backbone_after_zero_head_first_update():
    torch.set_num_threads(2)
    torch.manual_seed(37)
    model = NoFIRSonarMWCNN(8)
    features = torch.randn(1, 3, 24, 40)
    mask = torch.ones((1, 1, 24, 40), dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        (model(features, mask, 3)-.1).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.network.head[0].weight.grad.abs().sum() > 0


def test_whole_native_inference_is_independent_of_tile_partition():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    rng = np.random.default_rng(73)
    z = (rng.normal(size=(441, 777))+1j*rng.normal(size=(441, 777))).astype(np.complex64)
    z[:, :4] = 0
    model = NoFIRSonarMWCNN(4)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d) and layer.bias is not None:
                layer.bias.fill_(.02)
        model.network.tail.weight.normal_(0, .01)
    a = infer_raw(model, z, 'cpu', core_shape=(96, 256))
    b = infer_raw(model, z, 'cpu', core_shape=(192, 512))
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-4)
    assert a.std() > 0 and not a[:, :4].any()


def test_native_context_keeps_global_eight_pixel_lattice():
    box, inner = native_context((744, 12189), (311, 777, 96, 384))
    assert box[0] % 8 == box[1] % 8 == 0
    assert (box[0]+inner[0], box[1]+inner[1]) == (311, 777)


def test_sampler_never_uses_heldout_data_and_covers_all_training_domains():
    records = [dict(id=f'p{d}_{i}', role='public_train', domain=d) for d in (0, 1, 2) for i in range(7)]
    records += [dict(id=f'n{i}', role='native_train', domain=3) for i in range(8)]
    records += [dict(id='heldout', role='native_validation', domain=3),
                dict(id='heldout_public', role='public_validation', domain=0)]
    for phase in ('pretrain', 'adapt'):
        chosen = [plan_source(records, phase, i) for i in range(400)]
        assert not any('heldout' in item['id'] for item in chosen)
        assert {item['domain'] for item in chosen} == ({0, 1, 2} if phase == 'pretrain' else {0, 1, 2, 3})
        if phase == 'pretrain':
            assert {r['id'] for r in records if r['role'] == 'public_train'} <= {r['id'] for r in chosen}
    with pytest.raises(ValueError):
        plan_source(records, 'test', 0)


@pytest.mark.parametrize('domain', (0, 1, 2, 3))
def test_noise_recipe_preserves_original_support_and_does_not_filter_source(domain):
    torch.set_num_threads(2)
    z = torch.ones((24, 128), dtype=torch.complex64 if domain >= 2 else torch.float32)*.1
    z[:, :4] = 0
    before = z.clone()
    out, metadata = augment(z, domain, torch.Generator().manual_seed(37), profile())
    torch.testing.assert_close(z, before, rtol=0, atol=0)
    assert torch.isfinite(out).all() and not out[:, :4].any()
    assert not metadata['frequency_FIR']
    if domain < 2:
        assert not out.is_complex()


def test_source_guard_rejects_rewritten_admitted_array(tmp_path):
    path = tmp_path/'synthetic_test_array.npy'
    np.save(path, np.ones((12, 40), np.float32))
    stat = path.stat()
    item = dict(id='synthetic', domain=1, path=str(path), shape=[12, 40],
                file_guard=[stat.st_size, stat.st_mtime_ns])
    loader = Sources()
    assert loader.array(item).shape == (12, 40)
    np.save(path, np.ones((14, 40), np.float32))
    with pytest.raises(ValueError, match='source changed'):
        loader.array(item)


def test_sparse_positive_support_fallback_has_serializable_native_coordinates(tmp_path):
    import json
    path = tmp_path/'sparse_synthetic.npy'
    values = np.zeros((300, 900), np.float32)
    values[-1, -1] = .1
    np.save(path, values)
    stat = path.stat()
    item = dict(id='sparse_synthetic', domain=1, path=str(path), shape=[300, 900],
                file_guard=[stat.st_size, stat.st_mtime_ns])
    z, left, inner, core = Sources().context(item, 37, center=True)
    assert np.any(z > 0)
    assert all(type(x) is int for x in core)
    assert type(left) is int and all(type(x) is int for x in inner)
    json.dumps(dict(native_core=core, inner=inner, left=left))


def test_numpy_metadata_is_strict_JSON_and_nonfinite_does_not_replace_report(tmp_path):
    import json
    value = dict(native_core=[np.int64(2), np.int32(3)],
                 valid=np.bool_(True), values=np.array([1., 2.]))
    assert json.loads(json_text(value)) == dict(native_core=[2, 3], valid=True, values=[1., 2.])
    path = tmp_path/'metadata.json'
    save(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save(path, dict(value=np.float64('nan')))
    assert path.read_bytes() == before
