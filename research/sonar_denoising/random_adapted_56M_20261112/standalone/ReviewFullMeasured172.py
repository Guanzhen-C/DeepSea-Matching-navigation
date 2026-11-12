"""All frozen measured controls, unchanged native pixels and descriptive spectra.

The two candidate frequencies/areas were fixed in the earlier appearance
inspection, not fitted to this result. Their attenuation is NOT clean quality.
Log-domain row-median subtraction prevents a uniform darkening from being
misreported as removal of that periodic component.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from NativeSonarPaletteStream import amplitude_indices
from caris_orangecopper import caris_orangecopper
from TrainNativeLocalSonarWavelet import sha


STYLE_AREAS=(('port_strong',0,1728,9728,.109375,.0029296875),
             ('stbd_strong',1,768,9216,.16015625,.0029296875))
ORDER=('original','frozen_prior_anchored_raw','mandatory_wide_raw','fusion_wide_raw',
       'fusion_limited_raw','fusion_wide_front','fusion_limited_front')
LABELS=('原始实测','冻结旧网络','强制聚合／宽带','局部融合／宽带',
        '局部融合／限带','融合／宽带／前级','融合／限带／前级')


def projection(values,fy,fx):
    a=np.asarray(values,np.float64)
    if a.ndim!=3 or (a<=0).any() or not np.isfinite(a).all():raise ValueError('complete positive observed native patch required')
    _,h,w=a.shape;y=np.arange(h)[:,None];x=np.arange(w)[None]
    window=np.hanning(h)[:,None]*np.hanning(w)[None];q=np.log(a)
    q=q-np.median(q,axis=1,keepdims=True)
    return 2*np.abs(np.sum(q*window[None]*np.exp(-2j*np.pi*(fy*y+fx*x))[None],axis=(1,2)))/window.sum()


def region_values(a,field,coordinates):
    row,col,h,w=coordinates;bins=a.shape[-1]
    if col+w<=bins:side=0;radial=bins-col-w
    elif col>=bins:side=1;radial=col-bins
    else:raise ValueError('fixed area must lie on one physical side')
    first=row//4-1;phase=row%4;n=(phase+h+3)//4+1
    values=np.asarray(a[side,:,first:first+n,radial:radial+w])
    if field is not None:values=values*np.exp(np.asarray(field[side,:,first:first+n,radial:radial+w],np.float64))
    paired=(values[:4,1:]+values[4:,:-1]).transpose(1,0,2).reshape(4*(n-1),w)
    rows=paired[phase:phase+h]
    return rows[:,::-1] if side==0 else rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--voyage',type=Path,required=True);args=p.parse_args()
    root=args.voyage.absolute();out=root/'full_frozen_inference';start=time.monotonic()
    audit_path=out/'independent_native_audit.json';audit=json.loads(audit_path.read_text())
    if audit['status']!='passed_full_native_measured_transport_gain_mosaic_PNG_and_predeclared_neural_core_audit':
        raise ValueError('full native audit required before result review')
    path=out/'full_measured_quality_review.json'
    if path.exists():raise FileExistsError('preserve previous full control review')
    producer=json.loads((out/'run_summary.json').read_text());render=json.loads((out/'render_summary.json').read_text())
    manifest=json.loads((root/'metadata/measured_protocol_manifest.json').read_text())
    if sha(out/'run_summary.json')!=audit['producer_report_SHA256'] or sha(out/'render_summary.json')!=audit['render_summary_SHA256']:
        raise ValueError('audited producer/render records changed')
    a=np.load(root/'native_source/original_amplitude_float64.npy',mmap_mode='r')
    fields={name:np.load(out/f'{name}_supported_log_delta.npy',mmap_mode='r') for name in ORDER[1:]}
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',25)
    detail=Image.new('P',(7264,4476),0);detail.putpalette(palette.reshape(-1).tolist())
    preview=Image.new('P',(3892,1076),0);preview.putpalette(palette.reshape(-1).tolist())
    d=ImageDraw.Draw(detail);v=ImageDraw.Draw(preview);regions=[];diagnostics=[]
    original_regions=[region_values(a,None,r['coordinates']) for r in manifest['regions']]
    for j,(name,label) in enumerate(zip(ORDER,LABELS)):
        item=render['images'][name];file=Path(item['file'])
        if sha(file)!=item['file_SHA256']:raise ValueError('complete audited native image changed')
        Image.MAX_IMAGE_PIXELS=300_000_000
        with Image.open(file) as image:
            if image.mode!='P' or list(image.size)!=manifest['native_mosaic_shape'][::-1]:raise ValueError('unchanged native palette image required')
            image.load();v.text((j*556+8,22),label,font=font,fill=255)
            v.text((j*556+8,63),'全航段显示预览',font=font,fill=255)
            panel=image.resize((540,960),resample=Image.Resampling.NEAREST).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            preview.paste(panel,(j*556,116));d.text((j*1040+8,24),label,font=font,fill=255)
            record=dict(model=name,observed_regions=[])
            for i,r in enumerate(manifest['regions']):
                row,col,h,w=r['coordinates'];original=original_regions[i]
                numeric=original if name=='original' else region_values(a,fields[name],r['coordinates'])
                crop=image.crop((col,row,col+w,row+h))
                if not np.array_equal(np.asarray(crop),amplitude_indices(numeric)):raise ValueError('display source differs from native numeric region')
                yy=100+i*1094;detail.paste(crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM),(j*1040,yy+70))
                if j==0:d.text((12,yy+14),r['name']+'（原尺寸诊断，非干净真值）',font=font,fill=255)
                valid=original>1e-12;bright=valid&(original>=np.quantile(original[valid],.999))
                record['observed_regions'].append(dict(name=r['name'],
                    original_selected_bright_amplitude_ratio_median=float(np.median(numeric[bright]/original[bright])),
                    mean_positive_amplitude_ratio=float(numeric[valid].mean()/original[valid].mean())))
            regions.append(record)
    for label,side,row,col,fy,fx in STYLE_AREAS:
        original=np.asarray(a[side,:,row:row+256,col:col+1024]);before=projection(original,fy,fx);controls={}
        for name,field in fields.items():
            delta=np.asarray(field[side,:,row:row+256,col:col+1024],np.float64)
            changed=original*np.exp(delta);after=projection(changed,fy,fx)
            controls[name]=dict(before_log_component_by_beam=before.tolist(),after_log_component_by_beam=after.tolist(),
                median_component_amplitude_change_dB=float(np.median(20*np.log10(np.maximum(after,1e-30)/np.maximum(before,1e-30)))),
                mean_amplitude_ratio=float(changed.mean()/original.mean()))
        diagnostics.append(dict(label=label,side=side,native_region=[row,col,256,1024],
            fixed_frequency_per_record=fy,fixed_frequency_per_range_bin=fx,models=controls))
    detail_path=out/'真实完整航段_七列全部冻结模型_原尺寸细节_未验收.png'
    preview_path=out/'真实完整航段_七列全部冻结模型_全幅预览_未验收.png'
    detail.save(detail_path);preview.save(preview_path)
    result=dict(status='completed_full_measured_frozen_control_visual_and_descriptive_component_review',
                source_audit_SHA256=sha(audit_path),review_code_SHA256=sha(Path(__file__)),
                native_source_values=int(a.size),source_grid_or_inference_changed=False,
                original_palette_and_native_display_limits_unchanged=True,
                detail=str(detail_path),detail_SHA256=sha(detail_path),preview=str(preview_path),preview_SHA256=sha(preview_path),
                preview_only_uses_portrait_display_layout_not_native_resampling=True,
                regions=regions,fixed_candidate_component_diagnostics=diagnostics,
                frequencies_previously_used_for_noise_style_design_not_independent_holdout=True,
                component_attenuation_is_not_noise_rejection_ratio_or_clean_error=True,
                uniform_darkening_does_not_reduce_this_log_component_proxy=True,
                original_bright_returns_can_include_noise=True,clean_ground_truth=False,quality_accepted=False,
                elapsed_s=time.monotonic()-start)
    with path.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('regions','fixed_candidate_component_diagnostics')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
