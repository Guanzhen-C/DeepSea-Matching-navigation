# 三场景硬件在环仿真报告的研究任务与证据边界

目标是基于已有证据重新撰写研究报告，覆盖技术分析、算法验证、数据分析和结果解释，不写成安装记录或运行流水账。研究对象是多波束测深侧扫声呐、IMU、DVL及深度计条件下的海底地形地貌匹配导航，以及纯侧扫导航的板端部署验证。

## 一、阅读顺序

1. [完整原报告正文](REPORT.md)：算法、信息边界、地图、三组结果、尾部风险、SE3相对误差和实时性。
2. [已复核全量指标](data/verified_metrics.json)：三组DR、Local、Global的RMSE、分位数、最大值、末点值、十等分航程统计和尾部分析。
3. [数据字典](DATA_DICTIONARY.md)：坐标、通道、三维误差、航程定义及时间窗语义。
4. 海底审查：[完整性与未通过项目](evidence/sea_AGX/completion_and_integrity.json)、[SE3相对误差](evidence/sea_AGX/relative_motion_audit.json)、[时序与资源](evidence/sea_AGX/timing_and_risk.json)。
5. Chelan审查：[完整性与相对精度](evidence/lake_AGX/completion_and_integrity.json)、[SE3相对误差](evidence/lake_AGX/relative_motion_audit.json)。
6. 千岛湖NX：[全量独立核验](evidence/qiandao_NX/independent_full_audit.json)、[实际接收端诊断](evidence/qiandao_NX/receiver_summary.json)、[场景构建](evidence/qiandao_NX/scene_report.json)、[部署清单](evidence/qiandao_NX/deployment_manifest.json)。
7. [NX冻结算法源码清单](evidence/frozen_NX_algorithm_sources.json)：按清单阅读实际部署代码，不把当前仓库后来修改的代码当成历史运行版本。

## 二、算法分析重点

海底/Chelan为声学惯性特征前端与地图辅助混合后端，参考地图含测深和背散射。千岛湖采用特征式纯侧扫路线，局部GFTT/SIFT、全局LoFTR匹配，配合IMU/DVL/深度、回环和原生GTSAM固定滞后后端。千岛湖未消费测深或DVL离底高度，不能用海底版本的量测条件替换千岛湖条件。具体消费诊断优先于算法文件中某一功能的存在。

千岛湖核心源码：

- [前后端导航协调](algorithm_snapshot/qiandao_NX/standalone/PureSideScanFeatureNavigation.py)
- [特征地图](algorithm_snapshot/qiandao_NX/standalone/SideScanFeatureMap.py)
- [LoFTR特征](algorithm_snapshot/qiandao_NX/standalone/SideScanLearnedFeatures.py)
- [回环约束](algorithm_snapshot/qiandao_NX/standalone/SideScanLoopClosure.py)
- [斜距变化量测模型](algorithm_snapshot/qiandao_NX/standalone/SideScanMappedRangeChange.py)
- [斜距变化图优化](algorithm_snapshot/qiandao_NX/standalone/SideScanMappedRangeGraph.py)
- [实际板端接收执行](algorithm_snapshot/qiandao_NX/standalone/RunPureSideScanHILBoard.py)

需要论证的不是“是否叫FAST-LIVO2”，而是惯性误差传播、DVL约束方向、条带几何可观性、特征关联、声学视角变化、相关观测的信息预算、地图先验与评价真值的分离，以及全局约束怎样抑制长程漂移。

## 三、报告重写要求

建议围绕以下逻辑展开：研究问题与假设；传感器及环境模型；前端状态传播与声呐观测；全局候选、因子图及回环；三组实验设计与数据独立性；完整误差与分段/尾部/相对误差；实时性与部署约束；有效范围、不足与下一步可检验改进。

所有结论应区分实测事实、算法假设与待验证机制。平均误差下降不能证明所有时段稳定，峰值之后误差回落不能抹去失配区间，同航次不同记录不能证明跨航次泛化。不能把三组不同设备与不同输入的结果写成公平性能排名，也不能把仿真处理器在环写成真实AUV外场试验。

对“前端不足导致Global异常”“平坦区导致失配”“去噪提高定位精度”等解释，若缺乏同输入消融或对应事件证据，应写成机制假设及验证设计，而不是已证实的因果事实。没有干净声呐真值时，数值一致性不等于去噪质量证明。

报告日期与实验日期应依据记录真实标注，不把11月证据写成9月已完成。可以说明这些结果如何支撑初步板载部署与匹配算法仿真验证，但不能据此补造过去月份的验收或外场活动。

标题层级沿用：一级为题目，二级“一、”，下一级“1.”，再下一级“（1）”。优先采用原始论文、作者机构页面和官方数据说明核对外部技术主张；不能仅复述旧报告。

## 四、网页读取与计算限制

GitHub网页不直接展开完整GZIP/NPZ。若网页端无法读取这些二进制文件，先以JSON和报告为依据；需要独立重算时应下载完整仓库并运行verify_bundle.py，或让用户上传相应完整数据。必须说明自己是否实际读取并重算了数据，不能把阅读JSON写成重新处理全部轨迹。

*_error_preview_only.csv是有损抽样，仅帮助了解数据字段；没有保峰、保分位数或连续时长性质，禁止用它重算正式指标。源数据和凭据不在此公开包内，不要要求公开私有MAT来完成报告写作。
