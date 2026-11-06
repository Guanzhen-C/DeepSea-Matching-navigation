"""Complete native historical I/Q windows -> frozen denoiser -> SSS packets.

Only actual simulator-produced complex samples enter the neural model. No
phase is invented from an amplitude image at this boundary. Native samples
are preserved for inference; range pooling is only the navigation raster.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np

from PureSideScanFeatures import SideScanFeaturePacket


class FrozenComplexWindowDenoiser:
    def __init__(self, project, device='cuda'):
        import torch
        from FreezeSonarDenoisingRelease import verify
        from MatchedSonarWavelet import MatchedSonarWavelet
        project = Path(project)
        release = project/'releases/sonar_denoise_20261106'
        verified = verify(project, release)
        manifest = json.loads((release/'release_manifest.json').read_text())
        path = release/'weights/fusion_wide_front.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['condition_name'] != 'fusion_wide_front' or state['condition'] != dict(
                noise='wide', frontend=True, aggregation='learned_fusion'):
            raise ValueError('The user-pinned front/fusion function is required unchanged.')
        self.device = torch.device(device)
        self.model = MatchedSonarWavelet(state['width'], aggregation='learned_fusion').to(self.device)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.eval().requires_grad_(False)
        self.provenance = dict(model='fusion_wide_front',
            checkpoint_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            all_frozen_release_sources_and_weights_verified=True,
            release_source_count=manifest['source_count'],
            frozen_release_verification=verified,
            simulator_domain_transfer_NOT_previously_validated=True,
            preserves_complex_phase_via_positive_real_gain=True)

    def __call__(self, received):
        from MatchedSonarWavelet import infer_group
        received = np.asarray(received)
        if (received.ndim != 4 or received.shape[1:3] != (2, 8) or not np.iscomplexobj(received)
                or not np.isfinite(received).all() or received.shape[0] < 4):
            raise ValueError('Finite complete physical-ping x2-side x8-beam x native-range IQ required.')
        result = np.empty(received.shape, np.complex64)
        records = []
        for side in range(2):
            for beam in range(8):
                source = np.ascontiguousarray(received[:, side, beam], np.complex64)
                fields, info = infer_group({'fusion_wide_front': (self.model, True)}, source,
                    self.device, core_shape=(128, 1024))
                gain = np.exp(fields['fusion_wide_front'])
                result[:, side, beam] = source*gain
                records.append(dict(side=side, beam=beam, shape=list(source.shape),
                    finite_positive_gain=bool(np.isfinite(gain).all() and (gain > 0).all()),
                    neural=info['neural']['fusion_wide_front']))
        if not np.isfinite(result).all():
            raise ValueError('Frozen model generated nonfinite synthetic-domain output; do not silently substitute.')
        return result, dict(self.provenance, all_sixteen_native_beams_processed=True,
            physical_records=received.shape[0], native_range_samples=received.shape[-1],
            beams=records, future_real_pings_used=False, measured_heights_used=False,
            noise_free_quality_claim=False)


def packet_from_compressed_IQ(timestamp, compressed_IQ, parameters, *, bins=512):
    """Observed envelopes only; no world coordinates, clean IQ or height mask.

    A numerical floor rejects FFT roundoff. The declared receiver thermal
    floor rejects pure receiver-noise bins. Neither uses the clean signal.
    """
    import cv2
    z = np.asarray(compressed_IQ)
    if z.shape != (2, 8, parameters.range_bins) or not np.iscomplexobj(z) or bins < 2:
        raise ValueError('Native 2x8-beam complex pulse-compressed observation required.')
    power = abs(z).astype(float)**2
    amplitude = np.sqrt(cv2.resize(power.reshape(16, -1), (bins, 16), interpolation=cv2.INTER_AREA)).reshape(2, 8, bins)
    numerical_floor = max(float(amplitude.max())*1e-6, np.finfo(float).tiny)
    detection_floor = max(numerical_floor, 1.8*parameters.thermal_noise_rms)
    good = amplitude > detection_floor
    side, row, index = np.nonzero(good)
    ranges = (index+.5)/bins*parameters.max_range_m
    return SideScanFeaturePacket(timestamp=float(timestamp), ranges_m=ranges,
        amplitudes=amplitude[good], rows=row, sides=side, sample_sequence=index,
        row_centres_m=parameters.navigation_row_centres_m, max_range_m=parameters.max_range_m)
