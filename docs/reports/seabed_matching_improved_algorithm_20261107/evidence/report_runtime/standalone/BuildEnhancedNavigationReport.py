"""Rewrite the report only from a completed independently audited new run.

Retains applicable historical system/method/HIL sections but labels them as
historical; rewrites all current-scene results, interpretation and conclusion.
The original report is never modified. No raw data or uploads are performed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil

from BuildEnhancedNavigationComparison import load_json, save
from AnalyzeSideScanLoopReplay import sha


ROOT = Path(__file__).resolve().parents[1]
OLD_REPORT = ROOT / 'docs/reports/seabed_matching_simulation_20261107/海底地形地貌匹配算法仿真报告.md'


def section(text, title, body):
    pattern = re.compile(r'^## ' + re.escape(title) + r'\n.*?(?=^## |\Z)', re.M | re.S)
    result, count = pattern.subn(lambda _: '## '+title+'\n\n'+body.strip()+'\n\n', text)
    if count != 1:
        raise ValueError('Expected exactly one report section: '+title)
    return result


def change(new, old):
    ratio = (new-old)/old*100
    if abs(ratio) < 1e-12:
        return '不变'
    return ('增加' if ratio > 0 else '降低')+f'{abs(ratio):.2f}%'


def copy_referenced_images(text, destination, original):
    """No unreferenced historical/raw-looking previews in the new package."""
    destination, original = Path(destination).resolve(), Path(original).resolve()
    copied = []
    for relative in re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', text, re.M):
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination) or not relative.startswith('figures/'):
            raise ValueError('Report image must remain in its local figures directory')
        if target.exists():
            continue
        source = (original / relative).resolve()
        if not source.is_relative_to(original) or not source.is_file():
            raise ValueError('Referenced historical figure is missing')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    return copied


def write_report(directory, project=ROOT):
    directory = Path(directory).resolve()
    project = Path(project).resolve()
    original_report = project / OLD_REPORT.relative_to(ROOT)
    d = load_json(directory / 'data/verified_new_map_metrics.json')
    if d['status'] != 'completed_full_new_map_vs_unchanged_old_map_comparison' or d['samples'] != 62458:
        raise ValueError('Report requires the actual complete 62458-ping enhanced-map comparison')
    run = Path(d['source_run'])
    review = Path(d['source_review'])
    report = load_json(run / 'run_summary.json')
    if sha(run / 'run_summary.json') != d['new_sources']['executed_run_SHA256']:
        raise ValueError('Executed result changed since the independent comparison')
    target = directory / original_report.name
    if target.exists():
        raise FileExistsError('Do not overwrite previous report versions')
    old = original_report.read_text()
    m = d['metrics']
    ng, og, dr = m['new_Global']['rmse_m'], m['old_Global']['rmse_m'], m['DR']['rmse_m']
    current_change = change(ng, og)
    nav = d['new_graph']
    date = datetime.now().strftime('%Y年%m月%d日')
    preamble = f'''# 海底地形地貌匹配算法仿真报告

编制日期：{date}  
项目：基于迁移学习的海底地形重建与高精度定位技术研究  
本次修订：边缘修复与Shotwell增强新底图的完整梳形导航复验  
成果范围：PC新底图全程试验及旧海湖处理器在环历史验证  
原始声学数据、逐ping输入及可还原缓存：仅限本机，禁止上传

本报告以最新确认的去噪、边缘修复及暗场增强结果重新构建数值外观场，复用已验证的纯侧扫特征前端、来源投影、历史回环和混合后端，重新执行完整16.49 km梳形路线。新试验完整处理{d['samples']}个物理声呐ping；在线Global全程RMSE为{ng:.3f} m，DR为{dr:.3f} m，Global相对DR{change(ng, dr)}，相对旧底图相同算法的{og:.3f} m则{current_change}。这些数值来自全部在线输出，不是短程前缀、最后一次打印或最终离线平滑结果。

新版区分三类事实：新底图PC全程结果属于本轮实际复验；旧底图完整四组及海湖板载结果属于已归档历史证据；增强辐射映射属于非标定外观变化，不是新增测深精度或已证明的独立去噪增益。新旧参考场和仿真观测场均按各自版本生成；真值、航线和DR逐值一致，但声呐观测并不相同。本报告不要求新结果优于旧结果才披露，也不根据真实定位误差筛选或删除样本。

'''
    text = preamble + old[old.index('## 一、'):]
    # Unchanged method illustrations remain historical unless a real new-run
    # correspondence image is available, never passed off as a new observation.
    images = sorted((review).glob('实际侧扫地图特征_*.png'))
    if images:
        shutil.copy2(images[0], directory / 'figures/new_received_feature_match.png')
        text = text.replace('![直航与转弯侧扫子图](figures/qiandao_received_submaps.png)',
                            '![新底图实际查询子图及参考候选](figures/new_received_feature_match.png)')
        text = re.sub(r'^图2 .*$', '图2 本轮新底图实际接收回波形成的查询子图及参考候选。点是几何检查后的对应示例，不是人工标注真值；查询图不是从底图直接裁剪。', text, flags=re.M)
    else:
        text = text.replace('图2 当前方案的150 ping子图。', '图2 旧底图阶段保存的150 ping方法示例，并非本轮新观测。')
    text = text.replace('![整幅数据上的梳形规划](figures/qiandao_full_field_route.png)',
                        '![新数值外观场及完整梳形航线](figures/new_field_comb_trajectory.png)')
    text = re.sub(r'^图1 .*$', '图1 本轮新数值外观场及实际执行的完整梳形轨迹。该空间场由全部原生样本投影构建，不是将竖版瀑布图像素直接当作米制坐标。', text, flags=re.M)
    addition = '''### 5. 本轮新底图的定义

本轮不从RGB竖版图提取高度，也不把其展示纵横比当作地图比例尺。以原始MAT已核对的逐束幅度、固定网络增益和`gain_and_commonmode`边缘补丁为来源，在真实原生距离采样对应的几何支持样本上应用固定增强外观映射，再进行0.3 m空间格点平均。与旧场逐项核对后，三套参考、查询和显示场的高程、有效掩膜、观测计数及路线文件全部逐字节一致；改变的是强度外观。

增强映射采用原Shotwell 0.30.14输出的256项RGB调色板，以最大RGB分量除以255定义HSV-V标量代理。逐原生样本使用与原PNG相同的0至1截断及256级索引量化，然后查询映射表。该场不是标定背散射功率或物理回波幅度，空间格点平均也不同于原瀑布图8/4重叠显示。因此本报告使用“增强外观场”而不声称数值海图与竖版图逐像素同构。

参考与查询分别来自偶数、奇数物理记录，仍无共同记录ID；新场同时用于参考匹配和模拟新航线回波生成。不能只增强参考图而继续用旧观测冒充同一次新场仿真。由于回波场的外观幅度尺度发生变化，固定合成噪声下的相对信号水平也随之改变，结果差异不能单独归因于条纹去除。

'''
    text = text.replace('## 四、传感器模型与数据接口', addition + '## 四、传感器模型与数据接口')
    chapter6 = '''### 1. 原始复数处理与固定模型

设备提供的原始MAT属于波束形成复数记录，不是阵元ADC射频信号。按原MATLAB规则先对ULA1与ULA2复数求和，再求幅度，保留双舷每舷8束、全部2020物理记录及12189距离样点。固定神经模型为“融合／宽带／前级”第400步检查点，SHA256为`faece15dfd9c9a4b028a409b6d8ee583c93287e1b6ec81afd0c274c4686073e0`，没有为本次导航结果重新训练或改选模型。

前级采用采样率37650 Hz、−15 kHz至−3 kHz参考子带及257抽头滤波器。该频带仍是采集处理链假设，不是双舷射频完整标定。网络使用固定参考相位增量、幅度局部对比、两级Haar表示及学习的局部和方向共识融合，输出实数对数幅度增益；不预测或补造真实干净相位。模拟加噪训练与实测真实同帧干净标签不是一回事。

### 2. 采用后来确认的边缘修复

本轮显式选用后来保留的`gain_and_commonmode`，不是此前产生黑边被否决的候选。以末端1024原生列为上下文，保存末端512列增益补丁，实际修改末端384个有效列，256列平滑过渡；慢变估计按Data01与Data02分别计算，sigma为8条物理记录，不把两段边界混为连续慢变序列。

算法延续内部慢变增益，并去除边缘相对内部参考的额外时间共模分量。零填充尾部不涂黑、不补回波；空间坐标和原始MAT不改。此为统计辐射修正，不是已辨识的设备纯噪声或新神经推理，结构先验不成立的真实目标也可能受到影响。

### 3. 采用本机同版本Shotwell增强

使用Shotwell 0.30.14未修改的原始增强引擎。按GUI路径生成最长边360像素的分析图，实际为360×119；黑点0、白点161、阴影参数4.5913043。原生图仅更换256项RGB调色板，完整压缩空间索引流IDAT保持不变，原来纯黑的区域仍纯黑；没有生成式修图、额外锐化或全图缩小后重画。

完整增强PNG仍为24378×8084原生像素，高清竖版PDF内嵌全部样点，竖版仅改变展示矩阵。增强可能截断高亮、放大残余条纹；图更亮不证明噪声真值更小，也不证明任何导航精度必然改善。

![本次确认的增强结果预览](figures/approved_enhanced_portrait.png)

图0 本次确认的边缘修复与Shotwell增强全幅预览。预览不是原生分辨率文件；原生像素及来源哈希另有校验，私有数据仅本机保存。

### 4. 去噪与导航观测之间的边界

原生实测数据先处理形成新场；新的虚拟AUV观测再由该场和仿真器产生。导航收到的仍是含合成噪声的复数回波经脉冲压缩得到的幅度、IMU、DVL及压力深度，不是原瀑布图的直接裁剪，也不是重放真实原航次相位。历史ping只做普通重叠组织，不再次调用去噪网络。

新场的非标定HSV-V代理不应被解释为实物声呐绝对声压。参考图外观改变、边缘修复与合成信号水平改变同时存在，本轮对比评价的是这套组合方案，不是分别量化深度学习、小波、边缘修复和亮度增强的独立贡献。没有真实同帧干净真值，仍不报告真实PSNR或无损恢复率。
'''
    text = section(text, '六、实测侧扫去噪方案与效果边界', chapter6)
    chapter7 = '''### 1. 新旧底图对照

旧底图控制采用已完成的“自适应来源投影＋历史回环”最佳组合；新底图复用同一39文件冻结运行时、相同LoFTR公开室外权重、同一后端核心、准确初始状态、seed31和完整路线。高度、掩膜、有效计数及路线不改，不同时调前端或后端参数。新旧真值、时间戳和DR全程逐值核对一致，Local及Global允许因声呐外观变化而不同。

旧报告的2×2同输入消融证明表示和回环的作用，属于历史方法验证；本轮改变底图，必须重新生成声呐，不能称其与旧场具有完全相同声呐输入。两种设计的因果口径不同。本次只有一个完整新场噪声种子，不扩大为跨湖域或跨航次泛化证明。

### 2. 完整性与错误样本保留

新全程必须保存62458个声呐输出、1030546 IMU、103054 DVL及206109压力事件。单独50 m检查只验证启动和数据配对，不作为完整路线精度证据。独立审查实际CSV、NPZ、导航输入清单、源快照、参考地图及真实图因子；未完成或失败结果不填入最终报告。

### 3. 误差定义

每个共同采样时刻使用三维在线位置误差eₖ=‖p估计(tₖ)−p真值(tₖ)‖₂。RMSE为sqrt[Σeₖ²/N]，P95是95%样本不超过的分位值，不是置信区间；终点仅最后一个样本。全部样本保留，不作事后SE3对齐，不删除尖峰，也不以最终离线平滑替代实际在线Global。

相对运动使用10 m和30 m航程窗口的航向对齐平面RPE，不能与历史HIL按秒定义的SE3窗口混表。评价真值、真实误差和事后最坏段不进入匹配筛选或图因子构造。

### 4. 运行时效与历史在环条件

本轮是PC无界面完整执行，不是本次新底图的板载实时验收。物理ping周期约0.33 s，即约3.03 Hz，来自既定源时间戳；海湖历史在环分支使用5 Hz，属于不同条件。机器墙钟速度不改变模拟时间戳，不以是否快于实时替代定位效果和输入完整性。
'''
    text = section(text, '七、试验设计与评价指标', chapter7)
    text = text.replace('## 八、海湖处理器在环试验结果\n',
                        '## 八、海湖处理器在环历史试验结果\n\n本章沿用旧底图报告已独立核验的历史板载数据。本轮只在PC重跑千岛湖新外观场，未重跑这些板载试验，也未更新其历史成功或失败状态。\n')
    table = ['| 数据与通道 | RMSE/m | P95/m | 最大/m | 终点/m |', '|---|---:|---:|---:|---:|']
    for key, label in [('DR', '共同DR'), ('old_Local', '旧底图Local'), ('new_Local', '新底图Local'),
                       ('old_Global', '旧底图Global'), ('new_Global', '新底图Global')]:
        s = m[key]
        table.append(f'| {label} | {s["rmse_m"]:.3f} | {s["p95_m"]:.3f} | {s["max_m"]:.3f} | {s["final_m"]:.3f} |')
    worst = d['new_worst_Global']
    rpe = d['new_planar_relative']
    chapter9 = f'''### 1. 实际完整运行

本轮实际采样终点为{d['distance_m']:.6f} m，保留{d['samples']}条同步轨迹及全部规定事件。完整CSV与NPZ逐值一致，分析器独立重算指标；不是用200 m终端打印代替过程数据。参考与查询分别为1010条互不重复记录，源场仍具有名义航速与水平姿态假设。

### 2. 新旧底图全程误差

{chr(10).join(table)}

新Global相对共同DR的RMSE{change(ng, dr)}，相对旧底图Global则{current_change}。新Local相对共同DR的RMSE{change(m['new_Local']['rmse_m'], dr)}。不同指标分别解释，不以其中一个有利数值掩盖其它结果；终点更小也不能替代全程RMSE。

![新旧底图完整在线误差对照](figures/new_vs_old_full_errors.png)

图0 新旧底图全部在线误差。第一行保留DR及两种Local和Global，第二行单独展示两种Global，不删除异常峰。

### 3. 最坏区段与相对运动

新Global最大误差为{worst['error_m']:.3f} m，发生于累计{worst['distance_m']/1000:.3f} km。完整100 m分段统计另存CSV，不能只用平均精度掩盖转弯或弱约束区段。此位置为离线评价结果，不用于反向更改在线匹配门限。

新Global的10 m与30 m平面RPE分别为{rpe['10m']['RMSE_m']['Global']:.3f} m、{rpe['30m']['RMSE_m']['Global']:.3f} m；Local对应为{rpe['10m']['RMSE_m']['Local']:.3f} m、{rpe['30m']['RMSE_m']['Local']:.3f} m。绝对位置改善与短时发布连续性应分别评价，Global差分不是航行速度。

### 4. 实际地图先验和回环

新试验实际入图的侧扫绝对特征先验为{nav['map_priors']}条，历史回环因子为{nav['loop_factors']}条；旧底图同算法有{d['baseline_graph']['loop_factors']}条回环因子。数量来自实际GTSAM图与审查记录，不是检索候选数，也不是完整绕圈次数。是否增加约束不能单独证明精度提高。

### 5. 三维轨迹

![新底图三维地形与三条完整轨迹](figures/new_map_3d_trajectories.png)

图0 新底图完整真值、Local和在线Global轨迹。地形仍是原DOA与时延构建的仿真器私有世界，Z显示比例3倍；轨迹不人为错开、不贴地，不把显示地形传给纯侧扫估计器。

### 6. 历史消融的作用与本轮范围

旧底图四组Global RMSE分别为15.834 m、17.050 m、14.223 m、10.800 m，对应旧表示关/开回环、新表示关/开回环。该同输入结果用于确定本轮沿用的导航配置，并非本轮又跑了四组。新底图没有同时改变特征算法、回环逻辑或观测退化处理；新旧变化反映新的外观场及其产生的观测，不作为神经去噪独立增益证明。
'''
    text = section(text, '九、纯侧扫16.49 km完整仿真结果', chapter9)
    chapter10 = f'''### 1. 当前结果的解释

本轮新Global全程RMSE为{ng:.3f} m，旧底图为{og:.3f} m，即{current_change}。该结果按实际数值披露，不以视觉更亮为依据宣称导航更好。两组场景的几何和DR相同，说明差异主要发生在侧扫观测、特征关联和后端地图/历史约束链路，而不是更换了真实轨迹。

### 2. 外观增强不等于增加物理信息

Shotwell映射主要重新分配已有幅度的显示动态范围，没有恢复被滤除的信号，也没有产生新的测深或扫描方向。256级索引和高亮截断可能降低某些纹理的细粒度差别；提亮暗场同时会提亮残余噪声。原特征前处理本身已有局部辐射归一化与CLAHE，因此额外增强的作用必须由定位数据判断。

### 3. 可观性和后端传播

平行扫描的弱方向、未知地物高度与位姿耦合仍然存在。Local给出短时运动，参考地图与历史回环提供不同范围的约束；更好的外观对应不保证所有方向都有新的信息。错误匹配、发布修正与航向变化仍须联合检查，不能把Global绝对差分当作前端运动。

### 4. 不应扩大为实物精度或泛化结论

参考与查询来自同一航次的名义编码，虚拟梳形回波是合成观测，不是AUV真实执行了这些新航线。声速、外参、原航姿态和干扰模型仍未完整标定；本轮只有一个完整新场种子。结果用于验证该组合处理与导航链路，不是跨航次、未知初值失位恢复、真实声学标定或任意地形泛化验收。
'''
    text = section(text, '十、结果分析与工程适用性', chapter10)
    text = section(text, '十一、阶段完成情况与后续工作', '''### 1. 本轮完成的工作

完成全部原生数据新外观场构建、与旧几何逐字节对齐、同一冻结导航配置的50 m前置检查、完整16.49 km重新仿真、全部在线数据记录、独立图和数值审查、新旧全程对照及图文报告。50 m样本不代替长程，历史HIL不冒充本轮板载试验。

### 2. 后续应单独设置的对照

（1）分别设置固定去噪、仅边缘修复、边缘加增强，量化组合中各项作用，并对合成噪声相对尺度进行专门分析。

（2）补充独立航次、设备频带及外参标定、真实位置参考与时间同步证据，区分地图编码误差和导航误差。

（3）保持真实不可观方向，审查匹配错误与历史相关性，不通过增加虚假测深信息或硬反馈Global掩盖前端漂移。

（4）如开展板载部署或实时性验收，应以新完整配置重新运行，不能借用历史海湖完成记录认定当前纯侧扫新场已经通过。
''')
    text = section(text, '十二、软件与数据交付及复现方法', f'''### 1. 本地交付范围

本轮报告和完整派生定位结果保存在项目本地独立输出目录，旧正式报告不覆盖。原始MAT、I/Q、GPS、逐ping采集输入及可还原缓存禁止上传；本轮没有执行网络上传。完整声呐导航记录仍仅保留在本机实验目录，报告中的误差和虚拟轨迹也先作为本地交付。

### 2. 可复核数据

`data/full_errors_new_vs_old.csv`保留全部{d['samples']}个时刻的DR、旧/新Local及Global误差；`full_trajectories_new_vs_old.csv.gz`无损保存真值和五组完整估计坐标；`segments_100m_new_vs_old.csv`用于分段审查；`verified_new_map_metrics.json`保存重算指标、来源哈希、实际图边数与同DR核验。

`evidence/new_full_route_review.json`记录完整性和禁止观测项检查，`new_scene_report.json`记录新外观场来源。原版报告和已公开的代码发布是历史或软件资源，不因本次本地实验自动更新。

### 3. 执行配置

新试验使用固定39文件runtime、complex_IQ观测、LoFTR、adaptive_projected子图和历史回环，seed31、完整路线、无二次去噪。核心算法文件及权重不改；新场构建代码单独记录哈希。程序不依据评价真值调整量测门限或选择性接受试验。

### 4. 检查范围

新映射、原生几何、场景、路线及报告门禁、正文和打包逻辑的相关回归测试通过；此前冻结去噪95项文件哈希复核通过。新场高度、掩膜、计数和路线与旧场一致，独立分析按实际完整事件及全部轨迹重新计算指标。这些检查不代替跨设备标定、未知初始位姿恢复或实船精度验收。
''')
    text = section(text, '十三、结论', f'''新确认的边缘修复和Shotwell增强结果已转换为明确标注非标定性质的数值外观场，并完成全16.49 km匹配导航重新执行。全部{d['samples']}个在线输出和完整基础传感器事件得到保留和独立核验。新Global全程RMSE为{ng:.3f} m，相对DR{change(ng, dr)}，相对旧底图同算法{current_change}；最大误差{m['new_Global']['max_m']:.3f} m、终点{m['new_Global']['final_m']:.3f} m均如实列出。

本轮证明这套新数据处理、外观场、模拟声呐与已验证前后端能够完成有来源、有完整数据的导航试验，不预设亮度增强必然改善定位，也不把组合差异全部归于AI去噪。旧海湖处理器在环记录继续支持既有部署链路分析，但其历史跨架构与实时性限制没有被本次PC复验消除。

后续仍应以固定版本、独立航次及可分解的对照试验推进，保留弱方向、失败段与发布修正代价。原始声学数据和可还原缓存不上传，任何外部分享应另按授权范围处理。
''')
    text = re.sub(r'### 2\. 数据与本项目结果\n.*', '''### 2. 数据与本项目结果

本轮原始数据、处理输出及新旧场景核验仅在本机保留。新场来源和完整定位结果见本报告data与evidence目录；历史海湖和旧底图完整四组见原版报告及其归档证据，本轮不改写其实际执行日期和结果。

本轮增强采用GNOME Shotwell 0.30.14原始源码，见[固定版本源码及许可](https://github.com/Guanzhen-C/DeepSea-Matching-navigation/tree/sonar-denoise-enhanced-20261107/third_party/shotwell_0_30_14)。该引擎负责辐射外观映射，不是新增导航观测模型。
''', text, flags=re.S)
    approved = project / 'output/real_sonar/edge_shotwell_enhance_20261107_v2/Shotwell增强_竖版预览.png'
    shutil.copy2(approved, directory / 'figures/approved_enhanced_portrait.png')
    copied_historical_images = copy_referenced_images(text, directory, original_report.parent)
    counter = 0
    lines = []
    for line in text.splitlines():
        if re.match(r'^图\d+ ', line):
            counter += 1
            line = re.sub(r'^图\d+', '图'+str(counter), line)
        lines.append(line)
    text = '\n'.join(lines)+'\n'
    for image in re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', text, flags=re.M):
        if not (directory / image).is_file():
            raise ValueError('Report illustration missing: '+image)
    target.write_text(text)
    save(directory / 'evidence/report_rewrite_provenance.json', dict(
        original_report_SHA256=sha(original_report), rewritten_report_SHA256=sha(target),
        new_metrics_SHA256=sha(directory / 'data/verified_new_map_metrics.json'),
        original_report_unchanged=True, original_HIL_sections_are_historical=True,
        only_referenced_historical_figures_copied=copied_historical_images,
        figures=counter, source=str(target), original_data_uploaded=False))
    print(str(target), flush=True)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--project-root', type=Path, default=ROOT)
    a = p.parse_args()
    write_report(a.comparison, a.project_root)
