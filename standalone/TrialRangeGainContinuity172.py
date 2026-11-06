"""Two predeclared full-record boundary candidates, preserving frozen results.

All 16 physical beams and all 2020 records. Numeric gains are modified before
the original 4+4 mosaic; no RGB painting, cropping or neural retraining.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.recovery/'standalone'))
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from RangeGainContinuityRepair import repair, smooth_parts, PROTOCOL
    from FullMeasuredSonarProtocol import overlap_block, overlap_tail
    from NativeSonarPaletteStream import NativePaletteWriter, amplitude_indices, decoded_index_rows
    from caris_orangecopper import caris_orangecopper

    if a.out.exists():
        raise FileExistsError('Keep all previous candidates, including unsuccessful runs.')
    root = a.recovery/'measured_full_voyage_20261105_v1'
    old_root = root/'full_frozen_inference'
    audit = read(old_root/'independent_native_audit.json')
    render = read(old_root/'render_summary.json')
    producer = read(old_root/'run_summary.json')
    if (not audit['status'].startswith('passed_')
            or audit['producer_report_SHA256'] != sha(old_root/'run_summary.json')
            or audit['render_summary_SHA256'] != sha(old_root/'render_summary.json')
            or producer['models']['fusion_wide_front']['SHA256'] !=
               'faece15dfd9c9a4b028a409b6d8ee583c93287e1b6ec81afd0c274c4686073e0'):
        raise ValueError('Exact complete original approved-model audit required.')
    gain_path = old_root/'fusion_wide_front_supported_log_delta.npy'
    amp_path = root/'native_source/original_amplitude_float64.npy'
    old = np.load(gain_path, mmap_mode='r')
    amp = np.load(amp_path, mmap_mode='r')
    shape = (2, 8, 2020, 12189)
    if (old.shape != shape or amp.shape != shape or old.dtype != np.float32 or amp.dtype != np.float64):
        raise ValueError('Complete original F32 gains and F64 native magnitude source required.')
    parts = [x['count'] for x in read(root/'metadata/envelope_cache_summary.json')['parts']]
    if parts != [1033, 987]:
        raise ValueError('Original acquisition partitions required.')
    digests = {}
    for i in range(4):
        worker = read(old_root/f'worker_{i}.json')
        for item in worker['beams']:
            digests[item['side'], item['beam_0based']] = item['models']['fusion_wide_front']['applied_total_delta_SHA256']
    guards = {str(f): [f.stat().st_size, f.stat().st_mtime_ns] for f in (gain_path, amp_path)}
    code = {name: sha(Path(__file__).parent/name) for name in
            ('RangeGainContinuityRepair.py', 'TrialRangeGainContinuity172.py', '边缘修复试验协议.txt')}
    a.out.mkdir()
    names = ('gain_continuity', 'gain_and_commonmode')
    patches = {n: np.lib.format.open_memmap(a.out/(n+'_last512.npy'), mode='w+', dtype=np.float32,
                                           shape=shape[:-1]+(512,)) for n in names}
    started = time.monotonic()
    report_path = a.out/'run_summary.json'
    report = dict(status='running_full_records_gain_continuity_trial', protocol=PROTOCOL, PID=os.getpid(),
                  original_shape=list(shape), original_parts=parts, predeclared_candidates=list(names),
                  source_code_and_protocol_SHA256=code, source_guards=guards,
                  original_audit_SHA256=sha(old_root/'independent_native_audit.json'),
                  original_render_SHA256=sha(old_root/'render_summary.json'),
                  model=producer['models']['fusion_wide_front'],
                  statistical_gain_background_and_differential_commonmode_correction=True,
                  raw_IQ_filtered=False, new_neural_inference=False, training=False,
                  global_brightness_or_palette_changed=False, edge_gain_background_changed=True,
                  AI_generation=False, quality_accepted=False, adopted_as_default=False, beams=[])

    def save():
        report['elapsed_s'] = time.monotonic()-started
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        tmp.replace(report_path)

    def observe(amplitude, gains):
        v = amplitude[:, -512:]
        result = {}
        for name, d in gains.items():
            y = v*np.exp(d.astype(np.float64))
            edge = y[:, -141:-13]
            q = np.log(edge)
            centre = np.median(q, axis=1)
            hp = centre-smooth_parts(centre, parts)
            profile = np.median(y, axis=0)
            result[name] = dict(mean_edge_amplitude_vs_raw=float(edge.sum()/v[:, -141:-13].sum()),
                fraction_edge_amplitude_under_one_tenth_raw=float((edge < .1*v[:, -141:-13]).mean()),
                edge_commonmode_highpass_RMS=float(np.sqrt(np.mean(hp**2))),
                profile_edge_to_inner_ratio=float(np.median(profile[-269:-141])/np.median(profile[:128])),
                NOT_clean_error_or_noise_removal_rate=True)
        return result

    save()
    try:
        for side in range(2):
            for beam in range(8):
                original_gain = np.asarray(old[side, beam])
                h = hashlib.sha256(memoryview(np.ascontiguousarray(original_gain)).cast('B')).hexdigest()
                if h != digests[side, beam]:
                    raise ValueError('Full preserved baseline gain beam differs from original producer.')
                source = np.asarray(amp[side, beam, :, -1024:])
                present = source > 1e-12
                ends = shape[-1]-np.argmax(present[:, ::-1], axis=1)
                if not present.any(axis=1).all() or not np.all(ends == 12176):
                    raise ValueError('Explicit native endpoint changed.')
                values, meta = repair(original_gain[:, -1024:], source, parts,
                                      full_width=shape[-1], valid_end=12176)
                for name, value in values.items():
                    patches[name][side, beam] = value
                    patches[name].flush()
                metrics = observe(source, {'old_approved': original_gain[:, -512:], **values})
                report['beams'].append(dict(side=side, beam=beam, original_full_gain_SHA256=h,
                    source_magnitude_context_SHA256=hashlib.sha256(source.tobytes()).hexdigest(),
                    endpoint=12176, adapter=meta, observed_metrics=metrics))
                save()
                print(json.dumps(dict(completed_beams=len(report['beams']), total=16, metrics=metrics)), flush=True)
        report['status'] = 'completed_all16_all2020_gain_fields_pending_native_render'
        save()
        palette = (caris_orangecopper(256)*255).astype(np.uint8)
        height, width = 8084, 24378
        all_names = ('raw', 'old_approved', *names)
        previous = {n: None for n in all_names}
        writers = {n: NativePaletteWriter(a.out/('native_'+n+'.png'), width, height, palette,
                                           reserve_bytes=4*1024**3) for n in names}
        baseline = iter(decoded_index_rows(Path(render['images']['fusion_wide_front']['file']), palette))
        edges = {n: np.empty((height, 2, 512), np.uint8) for n in all_names}
        full_rows = np.rint(np.linspace(0, height-1, 1920)).astype(int)
        full_cols = np.rint(np.linspace(0, width-1, 1080)).astype(int)
        previews = {n: np.empty((1920, 1080), np.uint8) for n in all_names}
        count = checked = 0

        def consume(values):
            nonlocal count, checked
            size = len(values['raw'])
            indices = {n: amplitude_indices(v) for n, v in values.items()}
            for j in range(size):
                b = next(baseline, None)
                if b is None or not np.array_equal(indices['old_approved'][j], b):
                    raise ValueError('Independent complete frozen source rendering differs.')
                for n in names:
                    if not np.array_equal(indices[n][j, 512:-512], b[512:-512]):
                        raise ValueError('Candidate modified protected body pixels.')
                    checked += width-1024
            select = (full_rows >= count) & (full_rows < count+size)
            for n, data in indices.items():
                previews[n][select] = data[np.ix_(full_rows[select]-count, full_cols)]
                edges[n][count:count+size, 0] = data[:, :512]
                edges[n][count:count+size, 1] = data[:, -512:]
                if n in writers:
                    writers[n].append(data)
            count += size

        try:
            for start in range(0, shape[2], 16):
                stop = min(start+16, shape[2])
                original = np.asarray(amp[:, :, start:stop]).transpose(0, 2, 1, 3)
                restored = {'raw': original}
                for n in all_names[1:]:
                    d = np.array(old[:, :, start:stop], dtype=np.float64)
                    if n in patches:
                        d[..., -512:] = patches[n][:, :, start:stop]
                    restored[n] = original*np.exp(d.transpose(0, 2, 1, 3))
                values = {}
                for n, x in restored.items():
                    values[n], previous[n] = overlap_block(x, previous[n])
                consume(values)
            consume({n: overlap_tail(tail) for n, tail in previous.items()})
            if count != height or next(baseline, None) is not None:
                raise ValueError('Full native coverage incomplete.')
            report['native_images'] = {}
            for n, writer in writers.items():
                encoding = writer.finish()
                f = a.out/('native_'+n+'.png')
                report['native_images'][n] = dict(file=str(f), SHA256=sha(f), bytes=f.stat().st_size, encoding=encoding)
        finally:
            for w in writers.values():
                w.close()
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font = ImageFont.truetype(font_path, 27)
        small = ImageFont.truetype(font_path, 21)
        labels = ['原始含噪', '固定认可版', '仅增益连续', '增益连续＋相干条纹']
        panels = []
        for side, side_name in enumerate(('左舷', '右舷')):
            panel = Image.new('RGB', (4*536+16, 2050), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((16, 10), side_name+'边缘 · 全部2020记录（候选，待用户验收）', font=font, fill='black')
            draw.text((16, 48), '横向512原生列；纵向仅屏幕抽样；同琥珀色限；没有裁边或涂黑', font=small, fill='#444444')
            for j, (n, label) in enumerate(zip(all_names, labels)):
                x = 16+j*536
                draw.text((x, 88), label, font=small, fill='black')
                panel.paste(Image.fromarray(palette[edges[n][full_rows[::-1], side]]), (x, 130))
            f = a.out/(side_name+'_全航段边缘四方案对比.png')
            panel.save(f)
            panels.append(dict(file=str(f), SHA256=sha(f)))
        # Fixed native-pixel crops, including both ends and the middle. Not
        # selected retrospectively for good-looking regions.
        native_panel = Image.new('RGB', (4*536+16, 6*570+70), 'white')
        d = ImageDraw.Draw(native_panel)
        d.text((16, 10), '原生细节：4列同上，横纵均未缩放；三处固定区间 × 双舷', font=small, fill='black')
        for side in range(2):
            for k, top in enumerate((256, 3200, 6800)):
                y = 70+(side*3+k)*570
                d.text((16, y), ('左舷' if side == 0 else '右舷')+f' · 原生行 {top}–{top+512}', font=small, fill='black')
                for j, n in enumerate(all_names):
                    native_panel.paste(Image.fromarray(palette[edges[n][top:top+512, side][::-1]]), (16+j*536, y+40))
        detail = a.out/'双舷_三处原生细节_不缩放.png'
        native_panel.save(detail)
        overview = Image.new('RGB', (4*1100+16, 2050), 'white')
        d = ImageDraw.Draw(overview)
        d.text((16, 10), '全部2020记录：原始／固定版／两项边缘候选（显示抽样）', font=font, fill='black')
        d.text((16, 50), '两侧仅末端384有效列修改；其他区域逐像素核对不变；保持色限0–1', font=small, fill='#444444')
        for j, (n, label) in enumerate(zip(all_names, labels)):
            d.text((16+j*1100, 88), label, font=small, fill='black')
            image = Image.fromarray(palette[previews[n][::-1]])
            overview.paste(image, (16+j*1100, 130))
            if n in names:
                image.save(a.out/(n+'_仅屏幕预览.png'))
        overview.save(a.out/'全航段_四方案对比_仅屏幕预览.png')
        np.savez_compressed(a.out/'全原生边缘索引_全部8084行.npz', **edges)
        for f, stamp in guards.items():
            if [Path(f).stat().st_size, Path(f).stat().st_mtime_ns] != stamp:
                raise ValueError('Preserved source changed.')
        if any(sha(Path(__file__).parent/n) != h for n, h in code.items()):
            raise ValueError('Executed scientific code or declared protocol changed.')
        report.update(status='completed_all16_all2020_two_boundary_candidates_native_and_body_checked_NOT_quality_acceptance',
                      unchanged_body_pixels_checked=checked, original_native_PNG_pixels_replayed=height*width,
                      native_rows=count, edges=panels, detail=dict(file=str(detail), SHA256=sha(detail)),
                      full_source_and_weights_not_modified=True)
        save()
        print(json.dumps(dict(status=report['status'], elapsed_s=report['elapsed_s'], images=report['native_images']), ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=type(exc).__name__+': '+str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
