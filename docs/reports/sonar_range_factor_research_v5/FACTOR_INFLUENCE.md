# 斜距因子逐项贡献：观测、线性化和实际删边

本轮分析对象是历史千岛湖的 **304 个实际 GTSAM 批因子、3,431 对原始接收斜距**。M0 和阶段六批量 M1_Robust 使用同一批组织和同一序列 token；M1 的双距协方差形状仍是假设，不能因此声称两者似然相同。固定图按旧节点和真实 factor order 重建，不改变正式导航。

## 本地明细

明细（约 13 MB 压缩）含地图 XY、量距、相位中心与 token，仅在工作区的 `output/sonar_range_factor_research_v5/` 保留，不随 GitHub 发布。每行是一对历史仿真档案中记录的原始双斜距：

- $L$, 两个 $r_i$、接收器节点系偏移、两次垂向值、世界接收中心、两源 token、图边 ID、配对 ID、到达时间。
- 三组比较状态：档案初值；插边前的全历史 iSAM2 前缀代理；M0 和 M1 各自的全图后验。代理记录了用了哪些既有节点、上个优化边界、两边际协方差；**不是当时的真实导航预测**。
- M0 的剖面高度、其无鲁棒未预算的白化高度曲率、原差分量（m）、白化和正式加权鲁棒残差。严格等高 $Z$ 不可辨识则为 null，不能填 0 或最优唯一高度。
- M1 的可用高度最优值、曲率、Gaussian 半平方值、米制双距残差、白化残差及 Huber 后因子残差；不同于 M0 的标量差分。状态失败时高度和曲率不冒报有效，JSON null 表示未知。
- 两端各三维，采用 **GTSAM Pose2 局部切空间**。每一行保存 $J$ 和 $J^\top J$；它是加权鲁棒残差在该点的 GN 信息，不是单因子的独立传感器协方差，也不是当前位姿的边际精度。批内叠栈可求 6D 和条件目标 XY 奇异值，计算时旋转列以 20 m/rad 尺度归一。
- 实际 GTSAM LM 初值到第一次迭代调用返回后的两节点增量，以及初值到完整收敛返回的增量。它们是**全图**协同更新，不是把该批的独立梯度当作真实更新；精确省略该批后两端平移、全图位姿差另存在 C 收据。保留历史回环不变的斜距删除、同窗风险消融见[反事实实验](COUNTERFACTUAL_ABLATION.md)。

| 场景 | 逐对明细（本地） | 批级几何与摘要 |
|---|---|---|
| 16.49 km | `qiandao16_pair_detail.jsonl.gz`，3,119 行 | [诊断](evidence/qiandao16_diagnostic.json) · [影响](evidence/qiandao16_influence.json) |
| 4 km | `qiandao4_pair_detail.jsonl.gz`，202 行 | [诊断](evidence/qiandao4_diagnostic.json) · [影响](evidence/qiandao4_influence.json) |
| 2 km | `qiandao2_pair_detail.jsonl.gz`，110 行 | [诊断](evidence/qiandao2_diagnostic.json) · [影响](evidence/qiandao2_influence.json) |

[逐对计算](../../../research/sonar_range_factor_v5/diagnostics.py)调用保留的 [正式 M0](../../../standalone/SideScanMappedRangeChange.py#L93) 与 [M1 批剖面](../../../research/sonar_range_factor_v4/batch.py)，用簇内残差实际线性化尺度核对。图上的 LM/局部更新协议参考 [GTSAM 官方教程](https://gtsam.org/tutorials/intro.html)。每个公开诊断收据带对应私有明细的 SHA-256。

## 如何判断作用

符号定义 $\Delta={\rm RMSE}_{\rm drop}-{\rm RMSE}_{\rm full}$。$\Delta>0$ 表示该批在**给定其他约束和初值**时被保留更好；$\Delta<0$ 则删掉更好。阶段七不依据成绩重选因子。同一模型下残差 RMS 只是拟合冲突，不能判断其约束方向是否填补已有轨迹的薄弱方向，也不能判断地图、回环间的数据依赖。

- 16.49 km 的 277 个批，完整解残差 RMS 与带符号 $\Delta$ 的 Spearman 相关仅 **0.053（M0）/0.024（M1）**；与 $|\Delta|$ 则为 0.413/0.378，大小能反映部分敏感度但不能预测助益方向。完整解残差最大的是批 580，删后 M0 只变 −0.00030 m；相对地，删掉 M0 批 337 误差增 **0.10844 m**，它的残差并非最大。M1 最大帮助来自不同批 926，删后增 **0.09751 m**。共同坏批 950 删除后分别改善 0.03293/0.04024 m，但这仍不证明该组观测在任何网络中是错配。
- 4 km，M0 批 131 删除后增 **0.31842 m**，其完整图残差 RMS 仅 0.206；偏大的批 196 则删掉更好，两种模型中影响程度也不同。M1 的 5 次重跑失败如实保留，不用它们计算成功样本相关性。
- 斜距匹配依赖已接受地图关联；历史回环和地图先验的观测 **未保存逐样本 token**。故精确共用标志为 unknown；报告只追溯同到达时刻/同目标节点或同边端点的**关联风险**。归档中斜距自身的 token 是明确且唯一的，但不能据此排除与旧先验、同窗口的公共信息。
