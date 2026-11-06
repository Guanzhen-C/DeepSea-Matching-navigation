"""Detector-free LOCAL FEATURE matching; not optical flow or raw-image NCC.

LoFTR's public pretrained weights are unmodified. Sonar-specific physical
verification remains in SideScanFeatureMap: matches are not pose estimates.
No private sonar image is uploaded to obtain features.
"""
from pathlib import Path
import hashlib
import numpy as np


class LoFTRSideScanMatcher:
    def __init__(self, device='cuda', checkpoint=None):
        import torch
        import kornia
        from kornia.feature import LoFTR
        self.device = torch.device(device)
        path = Path(checkpoint) if checkpoint else Path(torch.hub.get_dir())/'checkpoints/loftr_outdoor.ckpt'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file('https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt',
                                           str(path), progress=False)
        self.model = LoFTR(pretrained=None).eval().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'], strict=True)
        self.provenance = dict(algorithm='LoFTR_local_features_plus_scan_plane_geometry',
            checkpoint_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            kornia_version=kornia.__version__, torch_version=str(torch.__version__),
            unmodified_public_outdoor_weights=True, sonar_specific_training=False,
            optical_flow=False, raw_intensity_correlation=False, input_images_uploaded=False)

    def __call__(self, query_image, prior_image, query_mask, prior_mask):
        import torch
        images = []
        for array in (query_image, prior_image):
            x = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None].to(self.device)/255.
            x = torch.nn.functional.pad(x, (0, (-array.shape[1])%8, 0, (-array.shape[0])%8))
            images.append(x)
        with torch.inference_mode():
            prediction = self.model(dict(image0=images[0], image1=images[1]))
        q = prediction['keypoints0'].cpu().numpy()
        m = prediction['keypoints1'].cpu().numpy()
        confidence = prediction['confidence'].cpu().numpy()
        valid = np.ones(len(q), bool)
        for points, mask in [(q, query_mask), (m, prior_mask)]:
            ix, iy = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)
            inside = (ix >= 0)&(iy >= 0)&(ix < mask.shape[1])&(iy < mask.shape[0])
            valid &= inside
            valid &= mask[np.clip(iy, 0, mask.shape[0]-1), np.clip(ix, 0, mask.shape[1]-1)] > 0
        return q[valid], m[valid], confidence[valid]
