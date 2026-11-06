"""Predeclare native review areas; render original/baseline/candidate pixels.

Fixed original CARIS limits and one output pixel per numeric source pixel.
These qualitative observation areas are not labelled clean seafloor targets.
No region is chosen by whichever candidate happens to look best there.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import hashlib
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PublicSonarNativeGrid import rgb_rows
from caris_orangecopper import caris_orangecopper


REGIONS = (
    ("Port 条带显著区", 6912, 1437, 1024, 1024),
    ("Stbd 条带显著区", 3072, 21405, 1024, 1024),
    ("结构观察区 1（非 clean 真值）", 6400, 6500, 1024, 1024),
    ("结构观察区 2（非 clean 真值）", 1000, 17000, 1024, 1024),
)


def verified_numeric_candidate(path):
    path=Path(path)
    report=json.loads((path.parent/'independent_native_and_stripe_audit.json').read_text())
    if report['status']!='passed_full_native_numerical_integrity_not_real_denoising_acceptance':
        raise ValueError('complete independent native audit required before detail rendering')
    expected=report['checked_numeric_output_sha256'].get(path.name)
    if expected is None:raise ValueError('candidate array was not part of the independent audit')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024**2),b''):digest.update(data)
    if digest.hexdigest()!=expected:raise ValueError('audited numeric candidate has changed')
    return expected


def original_pixels(array, region, palette):
    _, row, column, height, width = region
    if row<0 or column<0 or height<1 or width<1 or row+height>array.shape[0] or column+width>array.shape[1]:
        raise ValueError("declared native area is outside measured source grid")
    values=np.asarray(array[row:row+height,column:column+width])
    return rgb_rows(values[::-1],palette)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--original',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline-label',default='前一轮包络专用模型')
    parser.add_argument('--candidate-label',default='新模型候选（未验收）')
    parser.add_argument('--figure-name',default='固定原尺寸细节_原始_前轮_强度修正版_未验收.png')
    args=parser.parse_args()
    if args.declare_only:
        if args.protocol.exists():raise FileExistsError('preserve the first review-area declaration')
        args.protocol.parent.mkdir(parents=True,exist_ok=True)
        record=dict(declared_UTC=datetime.now(timezone.utc).isoformat(),regions=[list(x) for x in REGIONS],
            declared_before_current_candidate_measured_inference=True,
            native_mosaic_shape=[8084,24378],fixed_amplitude_display_limits=[0,1],
            is_clean_truth=False,automatic_quality_acceptance=False,
            purpose='Observe striping and strong structures without candidate-based ROI selection')
        args.protocol.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True);return
    if any(p is None for p in (args.original,args.baseline,args.candidate,args.output)):
        parser.error('original/baseline/candidate/output required for full native detail review')
    if Path(args.figure_name).name!=args.figure_name or Path(args.figure_name).suffix.lower()!='.png':
        raise ValueError('a plain PNG filename within the declared output directory is required')
    protocol=json.loads(args.protocol.read_text())
    checked_sha={str(path):verified_numeric_candidate(path) for path in (args.baseline,args.candidate)}
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    arrays=[np.load(path,mmap_mode='r') for path in (args.original,args.baseline,args.candidate)]
    if any(list(a.shape)!=protocol['native_mosaic_shape'] for a in arrays):
        raise ValueError('all candidates must retain the declared complete native grid')
    regions=protocol['regions'];width=1024;height=1024;gutter=16;header=100;band_header=70
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    canvas=Image.new('RGB',(3*width+2*gutter,header+len(regions)*(height+band_header)),'#151515')
    draw=ImageDraw.Draw(canvas)
    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(font_path),32) if font_path.exists() else ImageFont.load_default()
    titles=('原始实测（未滤波）',args.baseline_label,args.candidate_label)
    for index,title in enumerate(titles):
        draw.text((index*(width+gutter)+18,25),title,font=font,fill='white')
    descriptions=[]
    for index,region in enumerate(regions):
        name,row,col,h,w=region
        y=header+index*(height+band_header)
        draw.text((18,y+15),f'{name}  |  rows {row}:{row+h}, cols {col}:{col+w}',font=font,fill='white')
        for column,array in enumerate(arrays):
            image=original_pixels(array,region,palette)
            canvas.paste(Image.fromarray(image),(column*(width+gutter),y+band_header))
        descriptions.append(dict(name=name,row=row,column=col,height=h,width=w,
            input_pixels_per_output_pixel=1,display_interpolation_performed=False))
    path=root/args.figure_name;canvas.save(path)
    shutil.copy2(args.protocol,root/'predeclared_review_protocol.json')
    shutil.copy2(__file__,root/Path(__file__).name)
    report=dict(figure=str(path),size=list(canvas.size),areas=descriptions,display_column_labels=list(titles),
        original=str(args.original),baseline=str(args.baseline),candidate=str(args.candidate),
        independently_audited_numeric_sha256=checked_sha,
        fixed_original_palette=True,fixed_amplitude_limits=[0,1],resizing=False,
        original_source_files_written=False,clean_truth=False,quality_accepted=False,
        this_is_detail_review_after_full_native_inference_not_a_replacement_for_full_voyage_outputs=True)
    (root/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
