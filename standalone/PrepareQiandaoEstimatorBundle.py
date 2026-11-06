"""Export the unchanged validated lake estimator for the new PC-only scene."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.hil.frozen_engine import BASELINES, build_module
from standalone.hil.prepare_bundle import PATHS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    a = p.parse_args()
    scene = a.experiment/'scene'
    out = a.experiment/'estimator_bundle'
    if out.exists():
        raise FileExistsError('Keep earlier estimator bundles.')
    report = json.loads((scene/'scene_report.json').read_text())
    if report['GPS_file_opened'] or report['shared_record_IDs']:
        raise ValueError('GPS-free, physically disjoint scene partitions required.')
    source_commit = BASELINES['lake'][0]
    archive = subprocess.check_output(['git', 'archive', source_commit, *PATHS], cwd=ROOT)
    out.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for m in stream:
            if not (out/m.name).resolve().is_relative_to(out.resolve()) or not (m.isdir() or m.isfile()):
                raise ValueError('Unsafe archive entry.')
            stream.extract(m, out)
    _, audit = build_module(out/'standalone/large_scale_slam_isaac_demo.py', 'lake')
    (out/'map').mkdir()
    for f in (scene/'reference_baseline').iterdir():
        shutil.copy2(f, out/'map'/f.name)
    metadata = json.loads((out/'map/metadata.json').read_text())
    import numpy as np
    shape = np.load(out/'map/elevation_m.npy', mmap_mode='r').shape
    res = metadata['resolution_m']
    route = dict(schema='oceansim.measured-terrain-route/v1',
        map=dict(resolution_m=res, scene_extent_m=[(shape[1]-1)*res, (shape[0]-1)*res]),
        purpose='estimator map metadata only; actual simulator uses explicit virtual analytic trajectory',
        control_points_xy_m=[[-125., 30.], [125., 30.]],
        route_definition=dict(smoothing_iterations=0), GPS_used=False)
    (out/'map/full_map_traversal_route.json').write_text(json.dumps(route, indent=2)+'\n')
    manifest = dict(profile='lake', source_commit=source_commit, extraction=audit,
                    map_source='GPS-free small reconstructed Qiandao local scene, not Lake Chelan',
                    estimator_algorithm_changed=False, map_changed=True, files={})
    for f in sorted(out.rglob('*')):
        if f.is_file():
            manifest['files'][str(f.relative_to(out))] = dict(bytes=f.stat().st_size, sha256=sha(f))
    (out/'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='completed_unchanged_lake_algorithm_new_scene_bundle',
        files=len(manifest['files']), source_commit=source_commit, audit=audit)), flush=True)


if __name__ == '__main__':
    main()
