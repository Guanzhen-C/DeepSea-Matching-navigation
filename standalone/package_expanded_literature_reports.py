#!/usr/bin/env python3
"""Package only the eight checked report artifacts for the requested delivery."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import build_expanded_literature_reports as edition


def main():
    out = edition.OUT
    manifest = json.loads((out / 'evidence/交付核查.json').read_text())
    checks = json.loads((out / 'evidence/格式与范围复核.json').read_text())
    assert checks['status'] == 'pass'
    assert len(checks['reports']) == 4
    assert all(r.get('native_office_cjk_recovery') == 1.0 for r in checks['reports'])
    destination = Path('/home/cgz/Desktop/四份研究报告_纯文献深化版_20261002.zip')
    assert not destination.exists(), 'Preserve any existing delivery; choose a new archive name.'
    artifacts = []
    for item in manifest['reports']:
        for directory, suffix, field in [('word', '.docx', 'docx_sha256'), ('pdf', '.pdf', 'pdf_sha256')]:
            path = out / directory / (item['title'] + suffix)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == item[field]
            artifacts.append((path, item['title'][:3] + '/' + path.name, digest))
    assert len(artifacts) == 8
    note = ('四份独立文献研究报告：文献深化版\n'
            '整理日期：2026年10月2日。3月、5月、6月、7月为专题标签，不表示当月已完成该研究。\n'
            '每个专题文件夹均含可编辑Word及固定排版PDF，四份报告未合订。\n'
            f'PDF合计{manifest["total_pages"]}页，正文及文献合计约{manifest["total_cjk_characters"]:,}个汉字。\n'
            f'引用{manifest["unique_sources"]}项去重文献、标准及官方数据和软件资料，不全是论文。\n'
            '内容为公开文献综合研究、理论解释及建议的后续研究协议。\n'
            '不含具体项目的仿真结果、部署记录或自行完成的实验结果。\n'
            '各报告末尾列出参考文献及阅读范围；摘要资料不等同于全文核读。\n'
            '未附受版权保护的论文全文或完整译文。不能客观保证全网最详细。\n')
    with ZipFile(destination, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('阅读说明.txt', note)
        for path, name, _ in artifacts:
            archive.write(path, name)
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 9
        for _, name, digest in artifacts:
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    evidence = {'archive': str(destination), 'bytes': destination.stat().st_size,
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'report_files': 8, 'crc_check': 'pass', 'artifact_hashes': 'match',
                'entries': [name for _, name, _ in artifacts]}
    (out / 'evidence/压缩包核查.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
