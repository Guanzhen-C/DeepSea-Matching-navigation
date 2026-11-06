"""Display-only compact review of an already completed boundary experiment."""
from pathlib import Path
import argparse
import json

from PIL import Image, ImageDraw, ImageFont
from PackageBoundaryCandidatePortraitPDF import package
from ExportPinnedFullSamplePortrait import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', required=True, type=Path)
    a = p.parse_args()
    r = json.loads((a.trial/'run_summary.json').read_text())
    if r['status'] != 'completed_all16_all2020_two_boundary_candidates_native_and_body_checked_NOT_quality_acceptance':
        raise ValueError('Completed full-data trial required.')
    if len(r['beams']) != 16 or r['native_rows'] != 8084:
        raise ValueError('Full beam and record coverage required.')
    source = a.trial/'native_gain_and_commonmode.png'
    if sha(source) != r['native_images']['gain_and_commonmode']['SHA256']:
        raise ValueError('Exact new candidate required, not the old approved native image.')
    output = a.trial/'边缘修复候选_完整原生像素_高清竖版.pdf'
    pdf = package(source, output)
    fonts = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(fonts, 28)
    small = ImageFont.truetype(fonts, 22)
    comparison = Image.new('RGB', (4*536+16, 2050), 'white')
    d = ImageDraw.Draw(comparison)
    d.text((16, 10), '两侧边缘：固定认可版／本次修复候选，全部2020记录', font=font, fill='black')
    d.text((16, 49), '每列横向512原生列；纵向为全航段显示抽样；同色限，原固定结果保留', font=small, fill='#444444')
    for side, side_name in enumerate(('左舷', '右舷')):
        image_file = a.trial/(side_name+'_全航段边缘四方案对比.png')
        if sha(image_file) != r['edges'][side]['SHA256']:
            raise ValueError('Original completed comparison differs.')
        with Image.open(image_file) as panel:
            for j, (src_index, label) in enumerate(((1, '固定认可版'), (3, '本次修复候选'))):
                x = 16+(side*2+j)*536
                d.text((x, 88), side_name+' · '+label, font=small, fill='black')
                sx = 16+src_index*536
                comparison.paste(panel.crop((sx, 130, sx+512, 2050)), (x, 130))
    file = a.trial/'左右边缘_固定版与本次修复_全航段对比.png'
    if file.exists():
        raise FileExistsError('Preserve prior exports.')
    comparison.save(file)
    receipt = dict(report_SHA256=sha(a.trial/'run_summary.json'), source_SHA256=sha(source),
                   chosen_for_review='gain_and_commonmode', pdf=pdf,
                   comparison=dict(file=str(file), SHA256=sha(file)),
                   display_only=True, new_inference=False, quality_accepted=False, adopted_as_default=False)
    (a.trial/'高清交付校验.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
