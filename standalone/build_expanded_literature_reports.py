#!/usr/bin/env python3
"""Expand four independent literature reports without replacing the prior edition."""
from pathlib import Path
import json
import shutil

import build_literature_only_reports as report

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = report.OUT
OUT = ROOT / 'docs/project_reports/2026阶段研究_文献深化扩展版'
DESKTOP = Path('/home/cgz/Desktop/四份研究报告_文献深化扩展版')

ADDITIONAL_SOURCES = {
 'ttt': ('Zhang Q, Kim J', 'TTT SLAM: A feature-based bathymetric SLAM framework', 'Ocean Engineering, 2024, 294:116777. DOI:10.1016/j.oceaneng.2024.116777', 'https://www.sciencedirect.com/science/article/pii/S0029801824001148', '出版摘要、引言与可见方法片段 不声称全文复现', '测深特征与鲁棒SLAM直接研究'),
 'cadena': ('Cadena C, Carlone L, Carrillo H, Latif Y, Scaramuzza D, Neira J, Reid I, Leonard J J', 'Past, Present, and Future of Simultaneous Localization and Mapping: Towards the Robust-Perception Age', 'IEEE Transactions on Robotics, 2016, 32(6):1309–1332. DOI:10.1109/TRO.2016.2624754', 'https://arxiv.org/abs/1606.05830', '作者公开版本入口及摘要 用于SLAM问题分类', '机器人SLAM综述'),
 'gnc': ('Yang H, Antonante P, Tzoumas V, Carlone L', 'Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection', 'IEEE Robotics and Automation Letters, 2020. DOI:10.1109/LRA.2020.2965893', 'https://arxiv.org/abs/1909.08605', '作者摘要与公开版本入口 不据此声称无条件全局最优', '鲁棒估计方法基础'),
 'lie': ('Solà J, Deray J, Atchuthan D', 'A micro Lie theory for state estimation in robotics', '作者技术论文, 2018及修订版本, arXiv:1812.01537', 'https://arxiv.org/abs/1812.01537', '作者摘要及公开数学资料入口 本报告推导为独立说明', '状态估计数学基础'),
 'cello': ('Landry D, Pomerleau F, Giguère P', 'CELLO-3D: Estimating the Covariance of ICP in the Real World', '作者公开论文, 2018, arXiv:1810.01470', 'https://arxiv.org/abs/1810.01470', '作者摘要及公开论文入口 不作为水下实验结论', '数据驱动配准协方差参考'),
 'dip': ('Ulyanov D, Vedaldi A, Lempitsky V', 'Deep Image Prior', 'CVPR, 2018:9446–9454', 'https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html', '官方摘要与开放论文入口 仅讨论结构先验', '无外部训练数据的图像重建参考'),
 'spectral': ('Rahaman N, Baratin A, Arpit D, Draxler F, Lin M, Hamprecht F, Bengio Y, Courville A', 'On the Spectral Bias of Neural Networks', 'ICML, 2019, PMLR 97:5301–5310', 'https://proceedings.mlr.press/v97/rahaman19a.html', '官方摘要与论文入口 不推广为任意模型的普适结论', '神经网络频谱学习参考'),
 'cudem': ('NOAA NCEI', 'Coastal Elevation Models and Continuously Updated Digital Elevation Models', '官方沿海地形与水深数据产品说明', 'https://www.ncei.noaa.gov/products/coastal-elevation-models', '官方产品说明 具体数据精度需逐项元数据核查', '真实海底地形数据来源'),
 'gmrt': ('Global Multi-Resolution Topography synthesis', 'GMRT Overview and Technical Details', '官方多分辨率海底地形产品说明', 'https://gmrt.org/about/index.php', '官方产品与处理说明 不把全球底图视作全域高分测深', '真实海底地形数据来源'),
 'emodnet': ('EMODnet', 'EMODnet Bathymetry', '官方欧洲海域DTM、高分辨率产品与质量层说明', 'https://emodnet.ec.europa.eu/en/bathymetry', '官方产品说明 高分辨率限特定区域 原始测量获取条件另核查', '真实海底地形数据来源'),
 'aus': ('AusSeabed', 'Bathymetry data', '官方澳大利亚海底数据与测量覆盖说明', 'https://www.ausseabed.gov.au/data/bathymetry', '官方数据门户说明 不推断每个航次均含开放强度', '真实海底地形数据来源'),
 'shipwrecks': ('Sethuraman A V, Sheppard A, Bagoren O, Pinnow C, Anderson J, Havens T C, Skinner K A', 'Machine Learning for Shipwreck Segmentation from Side Scan Sonar Imagery: Dataset and Benchmark', 'The International Journal of Robotics Research, 2024在线出版. DOI:10.1177/0278364924126685', 'https://arxiv.org/abs/2401.14546', '作者摘要及作者数据网站 不将分割标签视为里程计真值', '真实侧扫数据与学习基准'),
 'subpipe': ('Álvarez-Tuñón O, Marnet L R, Antal L, Aubard M, Costa M, Brodskiy Y', 'SubPipe: A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization', '作者公开论文与官方数据集, 2024, arXiv:2401.17907', 'https://github.com/remaro-network/SubPipe-dataset', '作者论文摘要与官方数据说明 位姿参考来自导航估计而非无误差真值', '真实水下多模态序列数据'),
}

ADDITIONAL_FORMULAS = {
    'sr_footprint': [r'z_j=\sum_i A_{ji}h_i+\eta_j,\quad \sum_i A_{ji}=1', r'A_{ji}=\frac{\int_{\Omega_i}w_j(x,y)\,dx\,dy}{\int w_j(x,y)\,dx\,dy}'],
    'sr_dip': [r'\hat\theta=\arg\min_\theta\|M[\mathcal{D}(f_\theta(\xi))-z]\|^2+\lambda\mathcal{R}(f_\theta(\xi))'],
    'sr_interval': [r'\mathrm{Coverage}_{\alpha}=\frac{1}{N}\sum_i\mathbf{1}[h_i\in[L_{i,\alpha},U_{i,\alpha}]]'],
    'active_information': [r'\mathrm{EIG}(a)=E_{z\sim p(z|a,Z)}\!\left[D_{\mathrm{KL}}\!\left(p(H|z,a,Z)\|p(H|Z)\right)\right]'],
    'odom_jacobian': [r'r_i=n_i^T(Rp_i+t-q_i)', r'J_i=n_i^T\left[I\quad -R[p_i]_{\times}\right]'],
    'horizontal_dvl': [r'v_{xy}=R_z(\psi)v_b,\quad \delta v_{xy}\approx R_z(\psi)\delta v_b', r'+R_z(\psi)S v_b\delta\psi,\quad S(v_x,v_y)^T=(-v_y,v_x)^T'],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report.OUT, report.DESKTOP = OUT, DESKTOP
    report.SOURCES.update(ADDITIONAL_SOURCES)
    report.FORMULAS.update(ADDITIONAL_FORMULAS)
    # Rebuild from preserved original manuscripts; never append twice on reruns.
    shutil.copytree(PREVIOUS / 'manuscripts', OUT / 'manuscripts', dirs_exist_ok=True)
    (OUT / 'extensions').mkdir(exist_ok=True)
    for month in ('03月', '05月', '06月', '07月'):
        previous = (PREVIOUS / 'extensions' / (month + '专题分析.md')).read_text()
        added = (OUT / 'deepening' / (month + '深化研究.md')).read_text()
        (OUT / 'extensions' / (month + '专题分析.md')).write_text(previous + '\n\n' + added + '\n', encoding='utf-8')
    report.main()
    provenance = {'previous_edition': str(PREVIOUS), 'previous_artifacts_preserved': True,
                  'scope': 'independent_literature_synthesis_and_proposed_research_protocols',
                  'local_experiment_results': False, 'added_source_keys': list(ADDITIONAL_SOURCES)}
    (OUT / 'evidence/深化版来源说明.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
