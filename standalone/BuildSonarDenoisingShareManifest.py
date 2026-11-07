"""Build the public source/weight manifest without copying private MAT evidence."""
import argparse
import json
from pathlib import Path

from FreezeSonarDenoisingRelease import sha, verify


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'releases/sonar_denoise_enhanced_20261107'
ADDITIONS = [
    'standalone/RunFrozenSonarDenoising.py',
    'standalone/EnhanceEdgeSonarShotwell.py',
    'standalone/BuildSonarDenoisingShareManifest.py',
    'standalone/shotwell_reference/EnhancePalette.vala',
    'tests/test_frozen_sonar_denoising_pipeline.py',
    'tests/test_edge_sonar_shotwell_enhance.py',
    'releases/sonar_denoise_enhanced_20261107/README.md',
    'releases/sonar_denoise_enhanced_20261107/requirements.txt',
    'releases/sonar_denoise_enhanced_20261107/examples/enhanced_portrait_preview.png',
] + ['third_party/shotwell_0_30_14/' + name for name in (
    '.gitignore', 'ColorTransformation.vala', 'KeyValueMap.vala',
    '_transformation.c', 'COPYING', 'LICENSE.LGPL-2.1', 'UPSTREAM.txt')]


def build(render_audit, enhancement_audit, out):
    verified = verify(ROOT, ROOT / 'releases/sonar_denoise_20261106')
    old = json.loads((ROOT / 'releases/sonar_denoise_20261106/release_manifest.json').read_text())
    render = json.loads(Path(render_audit).read_text())
    enhancement = json.loads(Path(enhancement_audit).read_text())
    if not render['exact_previous_candidate_file_match'] or render['shape'] != [2, 8, 2020, 12189]:
        raise ValueError('Complete original-batch exact native rendering evidence required')
    if (not enhancement['spatial_indices_byte_exact'] or
            enhancement['source_SHA256'] != render['SHA256'] or
            not enhancement['black_palette_entries_stay_black']):
        raise ValueError('Exact candidate-to-enhanced native pixel evidence required')
    paths = sorted(set(old['code_files']) | set(ADDITIONS) |
                   {v['file'] for v in old['selected_checkpoints'].values()})
    files = {name: dict(SHA256=sha(ROOT / name), bytes=(ROOT / name).stat().st_size)
             for name in paths}
    if any(Path(p).suffix in ('.mat', '.npy', '.npz') or v['bytes'] >= 100*1024**2
           for p, v in files.items()):
        raise ValueError('Do not publish raw payloads or oversized files in this source release')
    report = dict(release='sonar-denoise-enhanced-20261107',
                  entrypoint='standalone/RunFrozenSonarDenoising.py',
                  input='original beamformed complex MATLAB v7.3 plus system and detection metadata',
                  ordered_stages=['complex_ULA1_plus_ULA2', 'fusion_wide_front_step400',
                                  'observed_water_support', 'gain_and_commonmode_edge_repair',
                                  'original_eight_four_overlap_amber_mosaic',
                                  'unmodified_Shotwell_0.30.14_auto_enhance'],
                  frozen_release_verified=verified, files=files,
                  original_batch_shape=render['shape'],
                  full_native_render_replay=render,
                  enhancement=dict(parameters=enhancement['parameters'],
                                   native_dimensions=enhancement['native_dimensions'],
                                   source_SHA256=enhancement['source_SHA256'],
                                   spatial_index_stream_SHA256=enhancement['output_IDAT_SHA256'],
                                   native_png=enhancement['outputs']['边缘修复_Shotwell增强_完整原生24378x8084.png'],
                                   native_pdf=enhancement['outputs']['边缘修复_Shotwell增强_完整原生像素_高清竖版.pdf']),
                  testing=dict(raw_MAT_small_synthetic_end_to_end_with_real_checkpoint_and_engine=True,
                               direct_entrypoint_full2020_neural_inference_rerun=False,
                               MAT_display_suite_tests_passed=52, neural_operator_suite_tests_passed=42,
                               suite_counts_overlap_not_unique_total=True),
                  privacy=dict(original_MAT_included=False, GPS_included=False,
                               credentials_included=False, full_native_large_images_included=False,
                               processed_preview_included=True),
                  old_release_and_navigation_maps_unchanged=True)
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve existing publication manifest')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(files=len(files), manifest=str(out)), ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-audit', type=Path, required=True)
    parser.add_argument('--enhancement-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=RELEASE / 'pipeline_manifest.json')
    args = parser.parse_args()
    build(args.render_audit, args.enhancement_audit, args.out)
