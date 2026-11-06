"""Lossless authorized-host receiver checks; synthetic data only."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys

import numpy as np


RECEIVER = Path(__file__).resolve().parents[1] / 'standalone/ReceiveNativeSonarArray.py'


def fixture_bundle(tmp_path):
    z = (np.arange(8*4*17).reshape(8, 4, 17) * (1+2j)).astype(np.complex128)
    source = dict(source_index=6, shape=list(z.shape),
                  source_guard=dict(size=1234, mtime_ns=1792982250313319001))
    bundle = dict(native_sources=[source], source_manifest_SHA256='a'*64)
    (tmp_path/'bundle_manifest.json').write_text(json.dumps(bundle))
    header = dict(source_index=6, native_shape=list(z.shape), dtype='<c16',
                  manifest_SHA256='a'*64, beams_0based=list(range(8)),
                  source_guard=source['source_guard'], native_context=[0, 0, 4, 17],
                  array_bytes=z.nbytes, array_SHA256=hashlib.sha256(z.tobytes()).hexdigest())
    return z, header


def receive(tmp_path, header, payload):
    encoded = json.dumps(header).encode()
    return subprocess.run([sys.executable, str(RECEIVER), '--bundle', str(tmp_path),
                           '--source-index', '6'],
                          input=struct.pack('>I', len(encoded))+encoded+payload,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_exact_payload_and_nanosecond_identity_survive_and_no_overwrite(tmp_path):
    z, header = fixture_bundle(tmp_path)
    result = receive(tmp_path, header, z.tobytes())
    assert result.returncode == 0, result.stderr.decode()
    folder = tmp_path/'native_arrays'
    final = folder/'source_06_complex128.npy'
    np.testing.assert_array_equal(np.load(final), z)
    receipt = json.loads((folder/'source_06_receipt.json').read_text())
    assert receipt['source_extractor_metadata']['source_guard']['mtime_ns'] == 1792982250313319001
    assert receipt['source_payload_SHA256'] == header['array_SHA256']
    original = final.read_bytes()
    assert receive(tmp_path, header, z.tobytes()).returncode != 0
    assert final.read_bytes() == original


def test_corrupt_or_truncated_payload_never_promoted_to_final(tmp_path):
    z, header = fixture_bundle(tmp_path)
    payload = bytearray(z.tobytes());payload[20] ^= 1
    result = receive(tmp_path, header, payload)
    assert result.returncode != 0
    assert not (tmp_path/'native_arrays/source_06_complex128.npy').exists()
    assert not (tmp_path/'native_arrays/source_06_receipt.json').exists()
    assert (tmp_path/'native_arrays/source_06_complex128.incomplete.npy').exists()


def test_wrong_source_manifest_rejected_before_any_payload_file(tmp_path):
    z, header = fixture_bundle(tmp_path)
    header['manifest_SHA256'] = 'b'*64
    assert receive(tmp_path, header, z.tobytes()).returncode != 0
    assert not list((tmp_path/'native_arrays').glob('*.npy'))
