import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from PublishJetsonHILResearchBundle import clean, clean_text, EVIDENCE


def test_private_fields_and_locations_redacted_without_changing_science():
    original = {'password': 'do-not-publish', 'rmse_m': 4.870919487986242,
                'source': '/media/test/private/path/trajectory.csv',
                'endpoint': '10.42.0.115', 'relative': 'standalone/example.py',
                'backend': 'gtsam_isam2_incremental_fixed_lag'}
    result = clean(original)
    assert result['password'] == '[redacted]'
    assert result['source'] == '[local-path]/trajectory.csv'
    assert result['endpoint'] == '[private-network]'
    assert result['rmse_m'] == original['rmse_m']
    assert result['relative'] == original['relative']
    assert result['backend'] == original['backend']
    assert original['password'] == 'do-not-publish'


def test_range_tokens_are_measurement_identity_not_credentials():
    result = clean({'used_distance_tokens': 7976, 'access_token': 'private',
                    'raw_MAT_uploaded': False, 'source_SHA256': 'a' * 64})
    assert result['used_distance_tokens'] == 7976
    assert result['access_token'] == '[redacted]'
    assert result['raw_MAT_uploaded'] is False
    assert result['source_SHA256'] == 'a' * 64


def test_report_terms_versions_and_public_sources_preserved():
    text = 'P95=10.462 m; CUDA 12.6; https://arxiv.org/abs/2104.00680'
    assert clean_text(text) == text
    assert all(Path(p).suffix == '.json' for p in EVIDENCE)
