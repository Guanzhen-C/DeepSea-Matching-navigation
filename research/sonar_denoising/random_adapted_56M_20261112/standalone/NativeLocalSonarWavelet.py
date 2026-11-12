"""New explicitly tagged finite-feature model; not v8/v9-compatible weights.

Uses the existing learned two-level Haar network and directional residual
consensus. The initial function is exactly identity. Only the learned residual,
never measured amplitudes, is spatially aggregated. No output gain floor,
global contrast stretch, post-hoc sharpening or observation clipping is used.
"""
import numpy as np
import torch
from torch.nn import functional as F

from PhaseAwareSonarWavelet import PhaseAwareSonarWavelet
from PublicEnvelopeCoherentModel import DirectionalNoiseConsensus
from LocalRelativeSonarFeatures import local_features, PROTOCOL


class SparseDirectionalNoiseConsensus(DirectionalNoiseConsensus):
    """Evaluate the same finite sparse kernels without dense 9x193 convolutions.

    All measured coordinates and kernel taps/mixture parameters remain the
    same. Floating-point summation order differs from the dense convolution;
    numerical forward/gradient equivalence is checked, not bit equality.
    """
    def __init__(self):
        super().__init__()
        self.taps=[]
        for kernel in self.kernels[:,0]:
            self.taps.append(tuple((int(y),int(x),float(kernel[y,x]))
                                   for y,x in torch.nonzero(kernel,as_tuple=False)))

    def forward(self,estimated_delta,support):
        if estimated_delta.ndim!=4 or estimated_delta.shape[1]!=1 or estimated_delta.shape!=support.shape:
            raise ValueError('aligned B1HW native predicted residual and support required')
        mask=support.to(estimated_delta.dtype)
        fields=torch.cat((estimated_delta*mask,estimated_delta.square()*mask,mask),dim=1)
        h,w=fields.shape[-2:]
        padded=F.pad(fields,(self.radius,self.radius,self.row_radius,self.row_radius))
        directions=[]
        for taps in self.taps:
            value=torch.zeros_like(fields)
            for y,x,weight in taps:
                value=value+padded[...,y:y+h,x:x+w]*weight
            directions.append(value)
        moments=torch.stack(directions,dim=1)
        first,second,weight=moments[:,:,0],moments[:,:,1],moments[:,:,2]
        denominator=weight.clamp_min(torch.finfo(estimated_delta.dtype).eps)
        mean=first/denominator
        variance=(second/denominator-mean.square()).clamp_min(0)
        mixture=torch.softmax(self.selector(torch.cat((mean,variance),dim=1)),dim=1)
        return (mean*mixture).sum(dim=1,keepdim=True)*mask


class NativeLocalSonarWavelet(PhaseAwareSonarWavelet):
    family = 'native_processed_finite_relative_scalar_wavelet_v1'
    feature_protocol = PROTOCOL
    # Includes the finite feature support. Conservative rectangular context;
    # factor-four origins are essential to the Haar sampling lattice.
    native_halo = (80, 208)

    def __init__(self, width=8):
        super().__init__(width=width)
        self.consensus=SparseDirectionalNoiseConsensus()
        with torch.no_grad():
            self.network.input.weight[:, 1:].zero_()
        for i in (0, 2):
            self.network.heads[i].requires_grad_(False)


def context_box(shape, core, halo=NativeLocalSonarWavelet.native_halo):
    h, w = shape; y, x, ch, cw = core; hy, hx = halo
    if (min(ch, cw) < 1 or min(y, x, hy, hx) < 0 or y+ch > h or x+cw > w
            or hy % 4 or hx % 4):
        raise ValueError('native core and factor-four real halo required')
    top = max(0, (y-hy)//4*4); left = max(0, (x-hx)//4*4)
    bottom = min(h, (y+ch+hy+3)//4*4); right = min(w, (x+cw+hx+3)//4*4)
    return (top, left, bottom, right), (y-top, x-left, ch, cw)


@torch.inference_mode()
def infer_native(model, values, device, *, use_phase, core_shape=(128, 1024)):
    """All native values, exact actual predecessor and factor-four context.

    Context features are derived from real native neighbors. Overlap/halo is
    discarded, not averaged into multiple differently normalized predictions.
    No full-voyage or private-data cache is persisted by this function.
    """
    z = np.asarray(values)
    if z.ndim != 2 or not np.iscomplexobj(z) or not np.isfinite(z).all():
        raise ValueError('one full native complex source required')
    result = np.empty(z.shape, np.float32)
    model.eval(); ch, cw = core_shape
    if min(ch, cw) < 4 or ch % 4 or cw % 4:
        raise ValueError('factor-four native inference core required')
    for y in range(0, len(z), ch):
        for x in range(0, z.shape[1], cw):
            box, core = context_box(z.shape, (y, x, min(ch, len(z)-y), min(cw, z.shape[1]-x)))
            top, left, bottom, right = box
            features, _, mask = local_features(z[top:bottom, max(0, left-1):right], first_column=left)
            f = torch.from_numpy(features[None]).to(device)
            support = torch.from_numpy(mask[None, None]).to(device)
            delta = model.feature_residual(f, support, use_phase=use_phase)[0, 0]
            cy, cx, height, width = core
            result[y:y+height, x:x+width] = delta[cy:cy+height, cx:cx+width].cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('nonfinite full native predicted residual')
    return result
