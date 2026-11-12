"""Actual-source augmentation qualification and small full-image contact sheet."""
from pathlib import Path
import argparse
import random

import numpy as np
import torch

from NoFIRSonarData import Sources, read, save
from RandomTriggeredSonarNoise import draw_scenario, apply_scenario, scenario_sha
from caris_orangecopper import caris_orangecopper


def main():
    from PIL import Image, ImageDraw, ImageFont
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    out = a.root / 'augmentation_qualification'
    out.mkdir(exist_ok=False)
    profile = read(a.root / 'random_ripple_profile.json')
    records = read(a.root / 'admission_manifest.json')['records']
    sources = Sources()
    canvas = Image.new('RGB', (4 * 480, 3 * 400 + 70), '#161616')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    draw.text((12, 12), 'Random lateral-source augmentation / actual public training data / original -> noisy (whole image)', fill='white', font=font)
    draw.text((12, 36), 'Fixed amber; native source pixels are not resized for training; images below are display previews only.', fill='white', font=font)
    palette = (caris_orangecopper(256) * 255).astype(np.uint8)
    cases = []
    for domain in range(4):
        pool = [r for r in records if r['domain'] == domain and r['role'].endswith('train')]
        for ordinal, item in enumerate(pool[:3]):
            original = np.asarray(sources.array(item))
            values = original.astype(np.complex64 if domain >= 2 else np.float32)
            seed = 5718000 + domain * 100 + ordinal
            scenario = draw_scenario(profile, values.shape, domain, seed, side=item.get('native_side'))
            gate = np.load(item['noise_support_file'], mmap_mode='r') if item.get('noise_support_file') else None
            observed = np.empty_like(values)
            for start in range(0, values.shape[0], 64):
                block, _ = apply_scenario(torch.from_numpy(values[start:start + 64].copy()).to(a.device),
                    profile, scenario, origin=(start, 0), support=None if gate is None else np.array(gate[start:start + 64]),
                    reference_RMS=item['global_reference_RMS'], beam=item.get('beam'))
                observed[start:start + 64] = block.cpu().numpy()
            y, x = max(0, values.shape[0] // 3), max(0, values.shape[1] // 3)
            h, w = min(100, values.shape[0] - y), min(400, values.shape[1] - x)
            cropped, _ = apply_scenario(torch.from_numpy(values[y:y + h, x:x + w].copy()).to(a.device),
                profile, scenario, origin=(y, x), support=None if gate is None else np.array(gate[y:y + h, x:x + w]),
                reference_RMS=item['global_reference_RMS'], beam=item.get('beam'))
            difference = np.abs(cropped.cpu().numpy() - observed[y:y + h, x:x + w])
            if not np.allclose(cropped.cpu().numpy(), observed[y:y + h, x:x + w], rtol=3e-6, atol=3e-7):
                raise ValueError('Actual full-image and training-context noise differ.')
            if gate is not None and not np.array_equal(observed[gate == 0], values[gate == 0]):
                raise ValueError('Protected native water changed in complete augmentation.')
            # One common display bound for original/noisy, not independent enhancement.
            limit = 1. if domain == 0 else float(np.quantile(np.abs(values), .995))
            for offset, array in enumerate((values, observed)):
                indices = (np.clip(np.abs(array) / max(limit, 1e-12), 0, 1) * 255).astype(np.uint8)
                image = Image.fromarray(palette[indices]).resize((235, 340), Image.Resampling.NEAREST)
                canvas.paste(image, (domain * 480 + offset * 240, ordinal * 400 + 110))
            draw.text((domain * 480 + 4, ordinal * 400 + 78), f'domain {domain} / seed {seed} / {len(scenario["events"])} sources', fill='white', font=font)
            cases.append(dict(id=item['id'], domain=domain, seed=seed, shape=list(values.shape),
                scenario=scenario, scenario_SHA256=scenario_sha(scenario),
                whole_crop_max_absolute_difference=float(difference.max()),
                protected_source_values_exactly_unchanged=True, original_reference_modified=False))
            print(dict(stage='qualified_complete_source', id=item['id'], domain=domain), flush=True)
    canvas.save(out / 'random_ripple_public_and_native_training_preview.png')
    save(out / 'run_summary.json', dict(status='passed_actual_source_whole_crop_noise_consistency',
        cases=cases, measured_clean_truth=False, full_source_native_pixels_used=True,
        training_display_resize=False, independent_preview_enhancement=False))


if __name__ == '__main__':
    main()
