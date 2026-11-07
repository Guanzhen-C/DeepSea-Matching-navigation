"""Rewrite the full local simulation report using completed improved results.

Keeps historical sea/lake HIL distinct from the new PC-only SSS algorithm.
Every current result comes from verified full-sample report evidence. Neither
original reports nor experiments are changed. No external upload capability.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

from BuildEnhancedNavigationReport import section
from BuildMappedRangeReportEvidence import BASE, PREVIOUS, ROOT, load, save
from AnalyzeSideScanLoopReplay import sha

TITLE='海底地形地貌匹配导航算法仿真报告'


def table(rows, heading=('方案或通道','RMSE / m','P95 / m','最大 / m','终点 / m')):
    result=['| '+' | '.join(heading)+' |','|'+'|'.join(['---']+['---:']*(len(heading)-1))+'|']
    for name, stats in rows:
        result.append('| '+name+' | '+' | '.join(f'{stats[k]:.3f}' for k in ['rmse_m','p95_m','max_m','final_m'])+' |')
    return '\n'.join(result)


def normal_stats(stats):
    return {key:stats[name] for key,name in [('rmse_m','RMSE_m'),('p95_m','P95_m'),('max_m','max_m'),('final_m','final_m')]}


def write_report(directory):
    directory=Path(directory).resolve(); d=load(directory/'data/verified_report_results.json')
    if d['status']!='completed_full_and_predeclared_fresh_report_evidence':
        raise ValueError('Completed audited report evidence required')
    original=PREVIOUS/'海底地形地貌匹配算法仿真报告.md'
    original_hash=sha(original); old=original.read_text()
    target=directory/(TITLE+'.md')
    if target.exists(): raise FileExistsError('Preserve previous documents')
    main,fresh=d['main'],d['fresh']; m,f=main['statistics'],fresh['statistics']
    mc,fc=main['comparison'],fresh['comparison']
    develop=d['development']; tempered=normal_stats(develop['tempered_full']['candidate_Global'])
    date=datetime.now().astimezone().strftime('%Y年%m月%d日')
    improvement=-mc['Global_RMSE_change_percent']; fresh_improvement=-fc['Global_RMSE_change_percent']
    relative_DR=100*(1-m['improved_Global']['rmse_m']/m['DR']['rmse_m'])
    preamble=f'''# {TITLE}

项目名称：基于迁移学习的海底地形重建与高精度定位技术研究  
编制日期：{date}  
修订内容：加入跨航线斜距变化约束的纯侧扫导航算法及完整对照验证  
成果范围：PC完整梳形导航、新噪声种子补充试验及历史海湖板载部署证据  
数据管理：原始声学数据、逐ping输入及可还原缓存仅限本机，禁止上传

本报告评价一种在既有声呐特征前端和混合后端上增加跨航线斜距变化约束的水下匹配导航方法。方法利用同一已关联地物的两次实际侧扫接收斜距，在不提供海底高度、测深似然或DVL离底高度的条件下，剖面化未知地物高度的物理可行区间，向GTSAM加入连接两个可变位姿的非线性约束。前端、航位推算、底图和传感器输入均保持不变，使整体定位变化主要对应新增后端约束及其引起的后续估计、关联变化。

在{mc['distance_m']/1000:.5f} km、{mc['samples']}个物理ping的完整同输入试验中，实际在线Global的三维位置RMSE从{m['baseline_Global']['rmse_m']:.3f} m降为{m['improved_Global']['rmse_m']:.3f} m，降低{improvement:.2f}%；P95从{m['baseline_Global']['p95_m']:.3f} m降为{m['improved_Global']['p95_m']:.3f} m，最大误差从{m['baseline_Global']['max_m']:.3f} m降为{m['improved_Global']['max_m']:.3f} m。全程终点误差则从{m['baseline_Global']['final_m']:.3f} m增加至{m['improved_Global']['final_m']:.3f} m，说明整体改善不等于各项指标均改善。

参数冻结后，另以未用于本轮调参的噪声种子生成完整{fc['distance_m']/1000:.5f} km、{fc['samples']}个ping的补充记录，原版与改进版使用同一记录。Global RMSE从{f['baseline_Global']['rmse_m']:.3f} m降为{f['improved_Global']['rmse_m']:.3f} m，降低{fresh_improvement:.2f}%；终点由{f['baseline_Global']['final_m']:.3f} m降为{f['improved_Global']['final_m']:.3f} m。该组沿航向及短距离相对误差略增，旧开发种子也保留轻微退化记录，结果不支持所有场景和所有指标全面提升的结论。

当前证据表明，跨航线重复观测中的真实斜距变化能够补充原纯侧扫方案薄弱的横向约束，而不是通过换用更清晰底图或虚构测深信息降低误差。相关117项测试及原海湖核心161项回归均通过。本次改进属于PC受控场景仿真；同航次名义地理编码、新航线合成回波、准确初始状态及近似相关性管理仍限定其解释范围。历史板载结果单列，不作为新算法已经完成板载实时验收的证明。

关键词：水下导航；侧扫声呐；地貌特征匹配；斜距变化；未知地物高度；因子图；同输入对照

'''
    text=preamble+old[old.index('## 一、'):]
    text=text.replace('本轮只在PC重跑千岛湖新外观场，未重跑这些板载试验',
                      '本轮在PC验证千岛湖纯侧扫后端改进，未重跑这些板载试验')
    text=text.replace('图2 本轮新底图实际接收回波形成的查询子图及参考候选。',
                      '图2 同输入基线阶段实际接收回波形成的查询子图及参考候选，作为表示方法示例而非本轮新增斜距因子示例。')
    text=text.replace('本阶段重点验证三类问题：', '本阶段总体验证三类问题：')
    intro_add='''### 4. 本次修订的核心验证问题

本次修订针对增强外观场上原方案Global RMSE仍为11.733 m的问题，验证新增后端物理约束是否能够有效利用重复侧扫观测，而非再次调整显示效果。对照保持底图、已记录的全部观测、初始化、局部前端和原地图及回环的既有信息强度不变；新增量测只使用确认特征、实际接收斜距及压力深度。评价覆盖完整路线、新噪声补充和已用于诊断的开发记录，分别考察平均误差、尾部、相对运动与不利结果。

'''
    text=text.replace('## 二、仿真系统总体方案',intro_add+'## 二、仿真系统总体方案')
    sensor_add='''### 5. 新斜距变化约束的数据来源

新增量测从导航实际收到的距离单元中取值，不从参考图的亮暗推断距离。直条带在原重叠平均之外附加主导实际接收来源标签，转弯来源投影继续使用原有标签；标签包含物理ping序号、同步沿航向通道、导航距离列和时间。附加标签不改变子图像素、描述子或Local更新。相同像素由多个回波贡献时，主导来源提供一个真实接收样本，而不是把平均图像对应的虚拟平均接收器当成实物声源。

距离列属于每舷512个导航单元，不应写为原生12189样点的全部独立距离精度。压力观测只用于航行器接收时刻的垂向状态；海底地物高度仍未知。当前后端采用平面位姿近似，传感器姿态、相位中心和外参误差的进一步标定仍会影响实物应用。

'''
    text=text.replace('## 五、导航与地形地貌匹配算法',sensor_add+'## 五、导航与地形地貌匹配算法')
    geometry='''### 6. 跨航线斜距变化约束

#### （1）从两次真实距离量测出发

设参考图上同一已确认地物的名义平面位置为L=(X,Y)，其真实垂向坐标Z未知。两次接收器的平面位置为oₐ、oᵦ，垂向位置为zₐ、zᵦ，实际接收斜距为rₐ、rᵦ。理想点散射几何满足rₐ²=‖L−oₐ‖²+(Z−zₐ)²，以及rᵦ²=‖L−oᵦ‖²+(Z−zᵦ)²。相减可得：

D = ‖L−oᵦ‖² − ‖L−oₐ‖² − (rᵦ²−rₐ²) = 2(zᵦ−zₐ)[Z−(zₐ+zᵦ)/2]。

该式来自同一地物两次球面距离的代数关系，不依赖图像横坐标被误认为水平距离。参考L提供名义平面锚定，但它本身仍有地理编码及关联误差。接收器偏移由Local运动和已知安装关系得到，两个端点的图位姿均作为优化变量，不把旧估计冻结成独立真值。

#### （2）等深消去与不等深区间

当zₐ=zᵦ时，未知Z在差分中精确消去，约束变为D=0；此时重复观察可能提供新的平面位姿信息。当接收深度不相同时，不能随意令Z为平底深度或从仿真地形读取高度。实现以最大斜距、下半空间和压力误差裕量构成保守的Z区间，再把该区间映射为D的允许区间[ℓ,u]。

残差写为ε=[D−clip(D,ℓ,u)]/[(rₐ+rᵦ)σ]。其中(rₐ+rᵦ)把平方距离差归一化为长度量纲，σ表示该归一化量的标准差。只要D落入允许区间，残差及相应雅可比均为零；未知高度允许的变化越大，得到的信息越弱。等深情况则保留真实等式在最优点的导数，不把它误作零信息死区。

本约束是保守区间兼容性模型，不等于完整联合地物最大似然解。它不能恢复所有地物高度，也不保证任意航线必然可观。两视角侧扫几何及平底假设的局限，可参见[两视角几何研究](https://arxiv.org/abs/2609.25271)；本报告的差分区间因子为本项目独立推导，未宣称完整复现该论文。

#### （3）可追溯的因果关联

只有已确认、可入图的地图特征对应能够进入历史关联库；候选基于参考平面坐标支持邻域检索，不使用真值误差选择。两次记录须来自不重叠物理窗口，并有至少180 s时间分离。每条因子至少6组特征、最多48组，超过上限时按距离支持分布抽取，不按真实定位误差排序。

每个距离样本用“物理ping、同步通道、导航距离列”唯一标识。一个样本在全部新增因子中最多使用一次；配对的旧端和新端均被消耗，不能在下一次回访时又当作新的独立观测。该限制只说明新增距离配对间不复用同一token，并不意味着它与旧地图、图像匹配或回环之间不存在相关性。

#### （4）噪声预算与鲁棒性

归一化噪声综合距离测量、地图平面位置及子图内部接收器偏移不确定性。当前工程配置的距离标准差为0.7 m、地图平面标准差2.5 m、接收器偏移标准差1 m、压力标准差0.03 m；这些是保守假设，不是该实物设备的外场标定结果。共享地图坐标在差分中的不确定性随两次接收基线传播，不把它计算为两份独立地图位置。

逐特征使用pseudo-Huber损失，参数为3；每条因子最多计24个独立等效特征，新增量测信息再乘0.5预算。鲁棒损失连续降低大残差影响，不以真实位置误差设定接受门限。窗口分离、特征数量及空间支持属于数据关联规则，不构成纯几何可观性的证明。

### 7. 图优化接入与策略选择

#### （1）真实两变量因子

新增因子使用GTSAM CustomFactor同时连接历史位姿和当前位姿，提供与Pose2局部增量约定一致的雅可比，并保留原固定滞后与长历史回环链路。存储格式中的零SE2占位矩阵仅用于兼容旧边容器；实际非线性距离数据另存，不能把占位矩阵误读为没有入图。

#### （2）保留既有信息的正式候选

最初试验把旧地图及回环的附加信息预算统一乘0.5，该策略在完整主场有改善，却在旧2 km开发记录上明显退化。退化早于首条新距离因子，说明统一减弱旧纠偏链路本身会改变结果。当前采用preserve_legacy：保留原地图及回环已经形成的信息矩阵，新增距离因子继续使用自身0.5预算，里程计不变。原本地图与回环共享当前窗口时的半信息分摊仍保留，没有取消后重复计入两份完整信息。

#### （3）输出与前端的边界

新约束只进入后端；DR和Local完整序列与基线逐值相同。后端通过既有map→odom软校正形成在线Global，不硬覆盖前端状态来隐藏漂移。新增因子引起优化状态变化后，后续检索、确认和地图先验数量也可能变化，因此实测改进反映整套闭环后端的效果，不能把所有收益机械分配给某一条距离因子。

#### （4）与公开方案的关系

[DIASSS公开实现](https://github.com/halajun/diasss)提供特征关联与侧扫几何优化的研究参考，但其演示接口包含离底高度和ground range等输入，本项目没有直接把这些量引入当前纯侧扫分支。[稠密子帧侧扫SLAM研究](https://arxiv.org/abs/2312.13802)说明可把对应子帧的相对约束组织为因子图；本项目使用的窗口、匹配器和未知高度兼容性模型仍以实际源码为准。

当前没有对所有共享地图、回环和距离关联建立严格联合协方差模型。一次性token、鲁棒损失和保守信息预算降低重复计数风险，但不等于已证明的协方差交集或统计一致性。未来应通过联合地物状态、相关性建模和独立外场验证进一步检验。

'''
    text=text.replace('## 六、实测侧扫去噪方案与效果边界',geometry+'## 六、实测侧扫去噪方案与效果边界')
    chapter7=f'''### 1. 主试验的同输入设计

主试验固定为增强外观场上已完成的完整16.49 km记录，噪声种子31。原版与改进版接收完全相同的{mc['samples']}个声呐ping、1030546个IMU、103054个DVL及206109个压力事件；不重新生成声呐。精确导航初始化沿用受控试验条件，参考图及核心依赖逐SHA核对。候选开启跨航线斜距变化约束，保持原地图和回环的既有信息强度。

此设计与旧报告的新旧底图对比不同：旧报告改变外观场并重新生成声呐，不能称为同声呐输入；本次把底图和收到的观测固定，主要考察后端对已有信息的利用。两种因果口径不混淆。

### 2. 开发记录与新噪声补充

旧场种子32的2 km记录用于诊断初版退化和选择保留既有信息的策略，属于开发验证，不再作为盲测。随后在新噪声记录生成前冻结参数和源码，使用种子33生成约4 km完整原版记录，再用候选重放并独立审查。两组运行串行，驱动核对91项预声明源哈希，防止看过新记录后再改参数。

新噪声只改变同一场景内的传感器随机实现，不构成独立航次、新海域或真实实船测试。新记录与主记录使用同一核心和同一参考图，均核对原始依赖哈希。补充样本数{fc['samples']}，不得替代主试验的62458样本全程结果。

### 3. 三维位置与方向误差

每个共同采样时刻计算eₖ=‖p估计(tₖ)−p真值(tₖ)‖₂，RMSE=sqrt[Σeₖ²/N]。P95为95%样本不超过的误差分位值，不是置信区间；最大误差保留全部异常峰；终点只评价最后一个时刻。数据按模拟采样时刻比较，不按网络到达时刻比较。

用评价真值航向ψ把水平误差投影为沿航向误差e∥=eₓcosψ+eᵧsinψ，以及横航向误差e⊥=−eₓsinψ+eᵧcosψ。真值航向只用于事后分解，不作为导航观测。该分解帮助辨认方向性薄弱环节，但不是单独证明因果归属或观测矩阵秩。

### 4. 相对运动与完整性评价

相对运动采用10 m和30 m航程窗口，将估计平面相对位移按各自起点航向转换后与真值相对位移比较。窗口内高度差不计入该平面RPE；它不能与历史板载按秒定义的SE3误差混表。所有时刻均保留，不作事后SE3对齐，不用最终全历史平滑轨迹替换实际在线Global。

完整性检查包括全部事件计数、归档输入哈希、CSV与NPZ逐值一致、实际图边、真实距离bin正支持、由Local重建的接收器偏移、压力位置及一次性token。改进版在首条新因子前的Global须与原版逐值相同，当前两组均有5549个前缀样本、坐标最大差0.0 m；该检查比较的是两份估计，不以真值误差筛选量测。

### 5. 运行时间与复现实验边界

本轮是PC离线加速的因果事件重放，观测频率来自原时间戳，声呐约3.03 Hz；海湖历史板载分支使用5 Hz。主候选重放墙钟约{main['extra']['elapsed_replay_s']:.1f} s，补充候选约{fresh['extra']['elapsed_replay_s']:.1f} s；这两项不含原生回波重新生成，不能与含生成阶段的总墙钟直接比较，也不能作为板载实时性的验收结论。
'''
    text=section(text,'七、试验设计与评价指标',chapter7)
    rows=[('共同DR',m['DR']),('共同Local',m['Local']),('原版Global',m['baseline_Global']),
          ('第一候选Global',tempered),('当前改进Global',m['improved_Global'])]
    direction=mc['candidate_directional']; prior_direction=mc['baseline_directional']
    chapter9=f'''### 1. 完整执行与主要结果

完整采样航程{mc['distance_m']:.6f} m，{mc['samples']}个在线输出及全部规定基础传感器事件齐全。原版、第一候选和当前保留既有信息的改进版使用相同收到的导航记录。DR和Local在三个后端方案中逐值相同，Global由各自实际在线发布值计算。

{table(rows)}

当前改进版相对原版Global的RMSE降低{improvement:.2f}%，相对共同DR降低{relative_DR:.2f}%。第一候选在完整主试验中也有明显改善，但它统一减弱旧信息并在开发记录上退化，不能仅凭主试验结果就优先采用。当前策略的平均、P95和最大误差进一步降低，而终点增大{m['improved_Global']['final_m']-m['baseline_Global']['final_m']:.3f} m，需与长程平均收益一起解释。

![主试验完整在线误差](figures/main_full_errors.png)

图0 同一62458样本输入的完整在线误差。上图同时保留DR、Local和两个Global；下图只比较原版与当前改进Global。所有异常峰均保留。

### 2. 方向性改善

| 方向RMSE | 原版 / m | 改进 / m |
|---|---:|---:|
| 沿航向 | {prior_direction['along_track_RMSE_m']:.3f} | {direction['along_track_RMSE_m']:.3f} |
| 横航向 | {prior_direction['cross_track_RMSE_m']:.3f} | {direction['cross_track_RMSE_m']:.3f} |
| 垂向 | {prior_direction['vertical_RMSE_m']:.5f} | {direction['vertical_RMSE_m']:.5f} |

原误差主要集中在横航向。新增真实斜距变化约束后，横向RMSE下降至{direction['cross_track_RMSE_m']:.3f} m，沿向也降低，垂向保持不变。这与新增约束用于平面后端而不是修改压力或前端的实现相符。该现象支持横向信息利用改善，但不能据此精确分配地图误差、匹配误差和融合误差各自占比。

![主试验沿向与横向误差](figures/main_directional_errors.png)

图0 采用评价真值航向分解的有符号误差。仅用于事后分析，不进入在线估计。

### 3. 首段与剩余峰值

首条新增斜距因子在模拟时刻{mc['preserved_prefix_audit']['first_new_range_timestamp_s']:.6f} s、累计{main['extra']['first_new_range_distance_m']/1000:.3f} km生效，此前5549个Global输出与原版完全相同。当前最大误差{m['improved_Global']['max_m']:.3f} m发生在累计{main['extra']['worst']['improved_Global']['distance_m']/1000:.3f} km，位于首次新增约束之前，因此本方法没有消除尚无独立回访的首段横向弱约束。

在首次新增约束后的全部样本中，原版Global RMSE为{main['extra']['after_first_range']['baseline_Global']['rmse_m']:.3f} m，改进版为{main['extra']['after_first_range']['improved_Global']['rmse_m']:.3f} m。该分段仅解释作用时序，不代替全程{m['improved_Global']['rmse_m']:.3f} m结果。原版最大峰位于{main['extra']['worst']['baseline_Global']['distance_m']/1000:.3f} km，其对应异常不通过删点或后处理掩盖。

### 4. 相对运动与发布代价

| 航程窗口 | 原Global平面RPE / m | 改进Global平面RPE / m |
|---|---:|---:|
| 10 m | {mc['baseline_planar_relative']['10m']['RMSE_m']['Global']:.3f} | {mc['candidate_planar_relative']['10m']['RMSE_m']['Global']:.3f} |
| 30 m | {mc['baseline_planar_relative']['30m']['RMSE_m']['Global']:.3f} | {mc['candidate_planar_relative']['30m']['RMSE_m']['Global']:.3f} |

主试验短距离Global相对误差略降，但仍须与Local连续里程计区分。map→odom发布修正可能使Global短时差分带有校正分量。用于控制或速度估计的短时运动应优先使用明确连续的里程计链路，而非把Global误差曲线的斜率当作航速。

### 5. 实际新增约束及回环

独立审查确认实际新增{mc['actual_range_input_audit']['factors']}条斜距变化因子、{mc['actual_range_input_audit']['feature_pairs']}组配对、{mc['actual_range_input_audit']['unique_received_samples']}个不复用距离token；传统回环{mc['actual_historical_loop_edges']}条、地图先验{mc['actual_new_map_priors']}条。全部新增观测均可追溯到实际接收距离单元及Local接收器偏移，未读取海底高度或评价真值构造因子。

![新增约束沿航程的实际使用](figures/actual_range_constraints.png)

图0 上图为真实入图的累计新增因子数，下图为各因子的配对特征数。因子数不是独立观测维数；信息仍受几何退化和相关性约束。

### 6. 改进后实际三维轨迹

![改进算法三维地形与三条轨迹](figures/improved_3d_trajectories.png)

图0 同一新底图上的真值、Local和改进后在线Global，分别用黑虚线、橙线和蓝线表示。每条轨迹保留62458个原采样位置，未人为平移、贴地或平滑。地形为仿真器私有世界，Z显示比例3倍；该高度场未送入纯侧扫估计器。
'''
    # Keep numerical expressions transparent in the saved narrative.
    text=section(text,'九、纯侧扫16.49 km完整仿真结果',chapter9)
    devbase=normal_stats(develop['preserved_development']['baseline_Global'])
    devold=normal_stats(develop['tempered_development']['candidate_Global'])
    devnew=normal_stats(develop['preserved_development']['candidate_Global'])
    chapter10=f'''### 1. 预先冻结的4 km新噪声检验

补充试验使用同一增强外观场、同一核心及参考图，但噪声种子改为33。原版完整生成并处理{fc['samples']}个ping后，候选直接重放其全部输入；没有根据中途误差继续调参。真实采样航程{fc['distance_m']:.6f} m，250000 IMU、25000 DVL、50000压力事件齐全。

{table([('共同DR',f['DR']),('共同Local',f['Local']),('原版Global',f['baseline_Global']),('改进Global',f['improved_Global'])])}

Global RMSE降低{fresh_improvement:.2f}%，P95降低，最大误差相同，终点明显降低。最大值仍由新约束前的共同首段决定，不应把最大不变解释为后续没有收益。该记录提供同场景新随机扰动下的补充证据，而不是完整16.49 km第二种子或跨海域泛化试验。

![新噪声补充试验完整误差](figures/fresh_full_errors.png)

图0 新种子完整15152样本同输入对照。原版与改进版首段完全重叠；后段整体漂移降低，但局部仍有改进版较差的区间。

### 2. 补充试验中的代价

横向RMSE由{fc['baseline_directional']['cross_track_RMSE_m']:.3f} m降为{fc['candidate_directional']['cross_track_RMSE_m']:.3f} m，沿向由{fc['baseline_directional']['along_track_RMSE_m']:.3f} m增至{fc['candidate_directional']['along_track_RMSE_m']:.3f} m。Global 10 m平面RPE由{fc['baseline_planar_relative']['10m']['RMSE_m']['Global']:.3f} m增至{fc['candidate_planar_relative']['10m']['RMSE_m']['Global']:.3f} m，30 m由{fc['baseline_planar_relative']['30m']['RMSE_m']['Global']:.3f} m增至{fc['candidate_planar_relative']['30m']['RMSE_m']['Global']:.3f} m。绝对误差改善与短时连续性并不等价，不能只选有利指标。

独立审查确认19条新斜距因子、202组配对、404个一次性token、47条传统回环和88条地图先验。首新因子前5549个Global坐标逐值相同，地图与核心依赖和主试验一致；输入全部无损复用，DR与Local不变。

### 3. 旧场开发记录及负结果

旧场种子32约2 km记录已参与诊断，不属于未见测试集。三种方案的完整记录结果如下：

{table([('旧场原版',devbase),('统一减弱旧信息候选',devold),('保留旧信息候选',devnew)])}

最初统一减弱旧信息的方案RMSE增加约22.35%，且退化先于首条新斜距因子；这一时序证据支持将旧信息减弱与新距离量测的作用分开。保留旧信息后，首新因子前轨迹恢复逐值一致，整体退化缩小到约0.98%，但终点仍差于原版。该记录不删除、不更名为成功结果，也不因为主试验改善就宣称已消除全部退化。

### 4. 能够支持的泛化范围

完整主试验与新噪声补充都支持当前场景下整体Global误差降低。它们不提供多地形统计显著性、独立航次绝对地图精度或任意失位恢复成功率。样本间存在强时间相关性，62458和15152个输出不等于同数量的独立实验；本报告不据逐帧样本量构造未经处理的置信区间或显著性检验。
'''
    text=section(text,'十、结果分析与工程适用性',chapter10)
    text=text.replace('## 十、结果分析与工程适用性','## 十、新噪声种子与开发验证结果')
    chapter11=f'''### 1. 11 m不是已证实的底图精度下限

同一底图和同一传感器记录下，Global RMSE从11.733 m降为5.040 m，说明原误差不能完全归因于图像质量或地图分辨率。原横航向误差占主导，新约束后横向明显改善，表明几何可观方向与后端信息利用是重要环节。地图0.3 m栅格间距并不等于0.3 m坐标精度；缺少外部测绘基准时，也无法给出真实绝对误差下限。

### 2. 去噪与定位改进的因果区分

本轮新旧算法均使用相同的固定去噪、边缘修复和增强外观场，收到的全部导航输入也相同。因此本轮57.04%的全程收益不能归因于又换了去噪模型、底图变亮或增加测深。旧报告新旧外观场的对比属于另一个试验，二者不混合计算所谓独立AI去噪增益。

### 3. 尚未解决的约束与统计问题

在没有独立重复地物的首段，新增斜距变化因子尚不可用，横向弱约束仍可能累计；未知地物高度和压力变化也会扩大允许区间、降低信息。需要通过具有共同地物的测线布设、交叉或反向复访设计及传感器标定增强真实几何，而不是增加不存在的高度观测。具体航线的最佳交叉角不能从单次实验概括为普适规则。

地图关联误差、多个图因子共享纹理及历史位姿相关性仍采用近似管理。即使新增token不复用，旧图像量测和新距离量测也不自动统计独立。后续可研究联合地物变量、相关性传播与输出连续性，但不能把这些未来工作写成当前实现已有的性质。

### 4. 月度目标的支撑情况

| 阶段目标 | 当前可支撑成果 | 解释范围 |
|---|---|---|
| 9月运行环境与板载部署链路 | 历史海湖完整板载接收、算法执行、定位返回与记录审查 | 初步部署链路，不等于全部实时性与跨平台精度通过 |
| 10月建立仿真环境 | 实际地形、声呐与运动传感器、梳形航线、记录和评价链路 | 本轮11月后续补充应按实际日期标注 |
| 绝对匹配算法仿真 | 历史多波束地形匹配及当前纯侧扫参考地貌匹配 | 当前5.040 m为同场受控仿真，不是独立实船绝对精度 |
| 相对匹配算法仿真 | 局部声呐特征里程计、历史回环、真实重复斜距约束 | Local仍有漂移，部分Global短时相对指标有退化 |
| 报告与复现资料 | 完整在线曲线、三维轨迹、全样本导出、配置和独立审查 | 原数据保密，仅本地交付 |

### 5. 下一阶段工作

优先针对首段横向弱约束和Global发布连续性开展独立对照，保留准确初始化与未知初值恢复两种不同评价；随后补独立航次、真实航姿和外参、声速与设备频带标定，并引入可靠外部定位参考。若在板卡部署新纯侧扫配置，应重新运行对应完整任务，而不能借用历史海湖结果认定新版本已通过。

原海湖核心及入口未改动，161项相关核心回归通过；这不等于重新执行了原72 km全程。当前改进作为显式可选策略保存，原版和所有正负试验继续保留，后续迭代可按相同输入、同样本和同指标回归。
'''
    text=section(text,'十一、阶段完成情况与后续工作',chapter11)
    text=text.replace('## 十一、阶段完成情况与后续工作','## 十一、结果分析与阶段成果')
    chapter12=r'''### 1. 本地交付与数据保密

本报告提供可编辑Word、PDF、源稿、全部主试验及补充误差CSV、无损压缩的完整仿真坐标、100 m分段统计、图表和来源摘要。原始MAT、实测I/Q、GPS、逐ping导航输入及可还原声学缓存不复制进报告包，也不上传。当前图表和仿真坐标仅作本地交付；任何对外分享仍应按数据授权范围单独审查。

### 2. 结果文件与数据字典

`data/main_errors_full.csv`保存62458行在线误差，`data/fresh_errors_full.csv`保存15152行。字段依次为模拟时间、累计航程、DR误差、Local误差、原版Global误差和改进Global误差，单位分别为秒、米、米。`main_trajectories_full.csv.gz`与`fresh_trajectories_full.csv.gz`保存相同采样时刻的真值及四组估计坐标，均为本地名义坐标，未丢弃样本。

`main_segments_100m.csv`与`fresh_segments_100m.csv`保留分段样本数、RMSE、中位数、P95、最大和末样本误差。分段用于定位异常区间，不是独立重复实验；跨段总体RMSE不能对分段RMSE不加权平均。`verified_report_results.json`保存当前重算结果、实际约束审查、开发负结果及回归范围。

### 3. 核心实现与版本绑定

核心新增模块分别负责未知高度区间残差、真实两变量GTSAM因子和一次性接收来源关联库。导航窗口只附加真实来源标签，前端保持原值。当前使用冻结42文件候选运行时，增强独立审查使用另存21文件分析运行时；4个原核心和3个参考图依赖逐SHA绑定，未用工作树中可能变化的文件替代已执行源码。

相关117项测试覆盖数学与雅可比、退化与物理区间、真实GTSAM接入、距离来源审计、默认关闭路径、完整性门禁、因子日志精度和报告依赖检查；原核心另161项回归通过。两份JUnit均0失败、0错误、0跳过。测试和清单只是验证的一部分，完整数值及实际图边仍需单独核对。

### 4. 同输入复现命令

以下命令使用本机已有私有场景和记录，不会下载、上传或覆盖原实验。输出必须选择从未存在的新目录。重放与分析须依次执行；这些命令用于复现已完成的同输入配置，本次编制报告没有重新启动仿真。

```bash
cd /home/cgz/OceanSim
TASK_BASE=output/real_sonar/qiandao_range_change_navigation_20261107_v1
TASK_SCENE=output/real_sonar/qiandao_edge_shotwell_navigation_20261107_v1
.venv-sonar-denoise/bin/python \
  "$TASK_BASE/runtime_candidate_v3/standalone/ReplayPureSideScanNavigation.py" \
  --experiment "$TASK_SCENE/enhanced_scene" \
  --run "$TASK_SCENE/full_enhanced_seed31" \
  --out "$TASK_BASE/manual_preserve_report_replay_NEW" \
  --runtime current --loop-closure \
  --submap-representation adaptive_projected \
  --mapped-range-changes --mapped-range-policy preserve_legacy
```

```bash
.venv-sonar-denoise/bin/python \
  "$TASK_BASE/analysis_runtime_v5/standalone/AnalyzeMappedRangeNavigation.py" \
  --source-run "$TASK_SCENE/full_enhanced_seed31" \
  --candidate "$TASK_BASE/manual_preserve_report_replay_NEW" \
  --out "$TASK_BASE/manual_preserve_report_review_NEW"
```

新增功能默认关闭；本报告采用显式的`--mapped-range-changes --mapped-range-policy preserve_legacy`。初版统一减弱旧信息策略只是保留的对照，不作为当前推荐命令。精确文件路径、源哈希和完整结果来源在evidence目录记录，正文不以工作目录标签代替算法定义。

### 5. 报告质量检查

报告数据再次从实际状态与在线CSV重算，检查样本总数、单调时间戳、所有坐标及误差逐值一致，并将标题、图、表、主要数值和PDF页面与源稿核对。Word标题遵循题目、一、1.、（1）的四级结构，图像保持宽高比。原版Word、PDF和源稿只读保留，新版单独存放。
'''
    text=section(text,'十二、软件与数据交付及复现方法',chapter12)
    text=section(text,'十三、结论',f'''当前纯侧扫改进利用同一地物两次真实接收的斜距变化，在未知地物高度下保持物理可行区间，通过两变量GTSAM因子补充原扫描平面与历史回环约束。完整16.49 km同输入试验的在线Global RMSE为{m['improved_Global']['rmse_m']:.3f} m，较原版降低{improvement:.2f}%，P95和最大误差也降低；同场景新噪声4 km补充的RMSE为{f['improved_Global']['rmse_m']:.3f} m，较其原版降低{fresh_improvement:.2f}%。全部规定事件及在线输出完整保留，DR、Local未变，实际距离来源和不复用条件通过独立审查。

改善具有明确边界：主全程终点误差从7.928 m增至9.940 m，补充测试沿向与短时相对误差略增，旧开发记录仍约0.98%轻微退化；首次独立回访前的弱方向尚未消除，历史相关性也未严格联合建模。因而结论是当前受控场景中的整体匹配导航性能改善，而不是任意场景或全部指标全面提升。

本报告可支撑现有仿真环境、绝对地形地貌匹配、相对匹配和部署链路阶段成果说明。历史海湖处理器在环结论保持原状；新纯侧扫算法本轮只在PC验证，仍需独立航次与外部基准、设备标定及对应板载全程复验，才能进一步形成真实高精度定位和实时时序验收结论。原始数据不上传，原算法与全部正负结果保留。
''')
    references='''### 1. 仿真与导航基础

（1）Song J, Ma H, Bagoren O, et al. OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework. IROS 2025; arXiv:2503.01074。[论文](https://arxiv.org/abs/2503.01074)

（2）Zheng C, Xu W, Zou Z, et al. FAST-LIVO2: Fast, Direct LiDAR–Inertial–Visual Odometry. IEEE Transactions on Robotics, 41:326–346. DOI:10.1109/TRO.2024.3502198。[出版页](https://ieeexplore.ieee.org/document/10757429/)；[官方代码](https://github.com/hku-mars/FAST-LIVO2)。本项目并非原样复现该激光视觉系统。

（3）Kaess M, Johannsson H, Roberts R, et al. iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree. International Journal of Robotics Research, 2012,31(2):217–236. DOI:10.1177/0278364911430419。[作者机构页面](https://publications.ri.cmu.edu/isam2-incremental-smoothing-and-mapping-using-the-bayes-tree)

（4）Sun J, Shen Z, Wang Y, Bao H, Zhou X. LoFTR: Detector-Free Local Feature Matching With Transformers. CVPR 2021:8922–8931。[论文](https://arxiv.org/abs/2104.00680)

（5）Zhang Q, Kim J. Terrain-Based Global Tracking Without Prior Pose Information Using Multibeam Echosounder Measurements. IEEE Transactions on Instrumentation and Measurement, 2026,75:7506113. DOI:10.1109/TIM.2026.3716492。[作者机构页面](https://pure.kaist.ac.kr/en/publications/terrain-based-global-tracking-without-prior-pose-information-usin/)

### 2. 侧扫特征与多视角几何

（1）Zhang J, Xie Y, Ling L, Folkesson J. A fully-automatic side-scan sonar simultaneous localization and mapping framework. IET Radar, Sonar & Navigation, 2024,18(5):674–683。[作者公开实现DIASSS](https://github.com/halajun/diasss)

（2）Zhang J, Xie Y, Ling L, Folkesson J. A Dense Subframe-based SLAM Framework with Side-scan Sonar. arXiv:2312.13802v2，2024年11月15日修订，作者页面标注为IEEE Journal of Ocean Engineering接收稿。[论文](https://arxiv.org/abs/2312.13802)

（3）Norman K, Mangelson J G. Beyond the Flat Seafloor: A Closed-Form Two-View Constraint to Aid Sidescan Sonar Reconstruction. arXiv:2609.25271，2026年9月21日，作者页面标注OCEANS 2026 Monterey接收。[论文](https://arxiv.org/abs/2609.25271)

上述文献提供仿真、特征关联、增量平滑及侧扫多视角几何基础，不提供本报告新增区间因子的性能保证。本报告所有定位数值来自本项目实际试验，不引用外部论文的误差数值替代自身验证。

### 3. 场景与软件来源

历史海区USGS Monterey Canyon数据来源为[10.5066/F7XD0ZQ4](https://doi.org/10.5066/F7XD0ZQ4)，历史湖区Lake Chelan数据来源为[10.5066/P1GE3YDR](https://doi.org/10.5066/P1GE3YDR)。千岛湖原始记录、固定去噪及本轮全部输入输出仅本机保留，见本报告的数据字典和来源摘要。

增强外观映射沿用GNOME Shotwell 0.30.14原引擎，源码与许可见[已固定的软件版本](https://github.com/Guanzhen-C/DeepSea-Matching-navigation/tree/sonar-denoise-enhanced-20261107/third_party/shotwell_0_30_14)。辐射增强不等于增加新的物理测深或导航观测。
'''
    text=section(text,'十四、参考文献与方法来源',references)
    number=0; lines=[]
    for line in text.splitlines():
        if re.match(r'^图\d+ ',line):
            number+=1; line=re.sub(r'^图\d+','图'+str(number),line)
        lines.append(line)
    text='\n'.join(lines)+'\n'
    for relative in re.findall(r'^!\[[^\]]*\]\(([^)]+)\)',text,re.M):
        if not (directory/relative).is_file(): raise ValueError('Missing figure '+relative)
    target.write_text(text)
    if sha(original)!=original_hash: raise ValueError('Original report must remain unchanged')
    save(directory/'evidence/report_rewrite_provenance.json',dict(
        previous_report=str(original),previous_report_SHA256=original_hash,
        current_report_SHA256=sha(target),evidence_SHA256=sha(directory/'data/verified_report_results.json'),
        original_report_unchanged=True,current_full_and_fresh_results_are_completed=True,
        historical_HIL_not_reexecuted=True,figures=number,external_upload_performed=False,
        writer_source_SHA256=sha(Path(__file__))))
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); print(write_report(args.report))
