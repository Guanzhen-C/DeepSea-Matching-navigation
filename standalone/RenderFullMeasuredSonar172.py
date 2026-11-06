"""Full-grid numeric mosaics and lossless original-palette PNGs for ALL models.

The main three-column preview is only a display layout; all native pixels and
all float32 log gain fields remain separately available at their full sizes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from FullMeasuredSonarProtocol import overlap_block,overlap_tail
from NativeSonarPaletteStream import NativePaletteWriter,amplitude_indices
from caris_orangecopper import caris_orangecopper
from TrainNativeLocalSonarWavelet import sha


def read(path):return json.loads(path.read_text())


def numeric_blocks(root,model_names):
    source=root/'native_source';out=root/'full_frozen_inference'
    a=np.load(source/'original_amplitude_float64.npy',mmap_mode='r')
    fields={name:np.load(out/f'{name}_supported_log_delta.npy',mmap_mode='r') for name in model_names}
    previous={name:None for name in ['original',*model_names]}
    for start in range(0,a.shape[2],16):
        stop=min(start+16,a.shape[2]);original=np.asarray(a[:,:,start:stop]).transpose(0,2,1,3)
        result={}
        for name in previous:
            restored=original if name=='original' else original*np.exp(
                np.asarray(fields[name][:,:,start:stop],np.float64).transpose(0,2,1,3))
            if not np.isfinite(restored).all() or (restored<0).any():raise ValueError('invalid unbounded native restoration')
            result[name],previous[name]=overlap_block(restored,previous[name])
        yield start*4,result
    yield a.shape[2]*4,{name:overlap_tail(tail) for name,tail in previous.items()}


def observation_changes(original,changed):
    valid=original>1e-12;cutoff=float(np.quantile(original[valid],.999));bright=valid&(original>=cutoff)
    return dict(original_only_bright_quantile=.999,original_only_bright_cutoff=cutoff,
                bright_observation_amplitude_ratio_q01_q50_q99=np.quantile(changed[bright]/original[bright],[.01,.5,.99]).tolist(),
                mean_positive_amplitude_ratio=float(changed[valid].mean()/original[valid].mean()),
                observed_bright_returns_may_include_interference=True,clean_truth=False)


def palette_image(indices,palette):
    image=Image.fromarray(indices,'P');image.putpalette(palette.reshape(-1).tolist());return image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--voyage',type=Path,required=True);args=p.parse_args()
    root=args.voyage.absolute();out=root/'full_frozen_inference';producer=read(out/'run_summary.json')
    if producer['status']!='completed_all_native_measured_log_gain_fields_pending_render_and_audit':
        raise ValueError('all complete frozen native gain fields required')
    manifest=read(root/'metadata/measured_protocol_manifest.json');receipt=read(root/'native_source/transfer_receipt.json')
    height,width=manifest['native_mosaic_shape'];names=['original',*producer['models']]
    main_names=manifest['main_comparison_models'];palette=(caris_orangecopper(256)*255).astype(np.uint8)
    writers={name:NativePaletteWriter(out/f'native_full_{name}.png',width,height,palette,reserve_bytes=32*1024**3) for name in names}
    digests={name:hashlib.sha256() for name in names};written=0;start_time=time.monotonic()
    preview_rows=np.rint(np.linspace(0,height-1,1280)).astype(int);preview_cols=np.rint(np.linspace(0,width-1,720)).astype(int)
    previews={name:np.empty((1280,720),np.uint8) for name in main_names};preview_seen=np.zeros(1280,bool)
    details={name:[np.empty(tuple(region['coordinates'][2:]),np.float64) for region in manifest['regions']] for name in main_names}
    seen=[np.zeros(region['coordinates'][2],bool) for region in manifest['regions']]
    report=dict(status='rendering_every_original_measured_mosaic_pixel_all_frozen_models',native_shape=[height,width],
                models=names,images={},source_code_SHA256=sha(Path(__file__)),
                source_protocol_manifest_SHA256=sha(root/'metadata/measured_protocol_manifest.json'),
                native_display_limits=[0,1],native_images_resized=False,
                preview_display_layout_is_sampled_not_added_measurement_resolution=True,quality_accepted=False)
    path=out/'render_summary.json'
    def save():
        report['elapsed_s']=time.monotonic()-start_time
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    save()
    try:
        for start,rows in numeric_blocks(root,producer['models']):
            if start!=written:raise ValueError('complete original native mosaic order required')
            size=len(rows['original']);selected=(preview_rows>=start)&(preview_rows<start+size)
            for name,values in rows.items():
                digests[name].update(memoryview(np.ascontiguousarray(values)).cast('B'))
                indices=amplitude_indices(values);writers[name].append(indices)
                if name in main_names:
                    previews[name][selected]=indices[np.ix_(preview_rows[selected]-start,preview_cols)]
                    for region,buffer in zip(manifest['regions'],details[name]):
                        top,left,h,w=region['coordinates'];begin=max(start,top);end=min(start+size,top+h)
                        if begin<end:buffer[begin-top:end-top]=values[begin-start:end-start,left:left+w]
            preview_seen[selected]=True
            for region,done in zip(manifest['regions'],seen):
                top,_,h,_=region['coordinates'];begin=max(start,top);end=min(start+size,top+h)
                if begin<end:done[begin-top:end-top]=True
            written+=size
            if written%640==0:
                report['native_rows_completed']=written;save();print(json.dumps(dict(native_rows=written,total=height)),flush=True)
        if written!=height or not preview_seen.all() or not all(done.all() for done in seen):
            raise ValueError('all native pixels and fixed diagnostic regions required')
        for name,writer in writers.items():
            encoding=writer.finish();file=out/f'native_full_{name}.png'
            report['images'][name]=dict(file=str(file),file_SHA256=sha(file),bytes=file.stat().st_size,
                                       encoding=encoding,float64_mosaic_stream_SHA256=digests[name].hexdigest())
        original=report['images']['original']
        if (original['encoding']['index_SHA256']!=receipt['trailer']['original_forward_mosaic_index_SHA256']
                or original['float64_mosaic_stream_SHA256']!=receipt['trailer']['original_forward_float64_mosaic_SHA256']):
            raise ValueError('complete original display and float64 mosaic must reproduce local source exactly')
        fontpath=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        font=ImageFont.truetype(str(fontpath),25)
        labels=['原始实测（非干净真值）','冻结旧网络（全航段）','新宽带／强制聚合（全航段）']
        canvas=Image.new('P',(2192,1396),0);canvas.putpalette(palette.reshape(-1).tolist());draw=ImageDraw.Draw(canvas)
        for j,name in enumerate(main_names):
            draw.text((j*736+12,24),labels[j],font=font,fill=255)
            draw.text((j*736+12,66),'全航段显示预览；非原生文件',font=font,fill=255)
            canvas.paste(palette_image(previews[name][::-1],palette),(j*736,116))
        preview=out/'完整航段_原始_冻结旧版_新模型_琥珀色预览_未验收.png';canvas.save(preview)
        canvas=Image.new('P',(3104,4476),0);canvas.putpalette(palette.reshape(-1).tolist());draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(labels):draw.text((j*1040+12,24),label,font=font,fill=255)
        changes=[]
        for i,region in enumerate(manifest['regions']):
            yy=100+i*1094;draw.text((12,yy+14),region['name']+'（每格1024×1024原尺寸）',font=font,fill=255)
            for j,name in enumerate(main_names):canvas.paste(palette_image(amplitude_indices(details[name][i][::-1]),palette),(j*1040,yy+70))
            changes.append(dict(**region,models={name:observation_changes(details['original'][i],details[name][i]) for name in main_names[1:]}))
        detail=out/'完整航段推理_固定原尺寸细节_原始_旧版_新版_未验收.png';canvas.save(detail)
        report.update(status='completed_all_seven_native_images_and_fixed_original_pixel_diagnostics_not_quality_acceptance',
                      native_rows_completed=written,original_numeric_mosaic_and_display_exact=True,
                      original_native_mosaic_values=height*width,preview=str(preview),detail=str(detail),
                      predeclared_observation_changes=changes)
    except BaseException as exc:
        report.update(status='failed_preserved_measured_render_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        for writer in writers.values():writer.close()
        save()


if __name__=='__main__':main()
