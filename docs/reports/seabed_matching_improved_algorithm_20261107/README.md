# 最新匹配导航算法与报告阅读入口

本目录对应加入跨航线斜距变化约束的纯侧扫导航改进，包含已完成的完整16.49 km同输入对照和参数冻结后的4 km新噪声补充。报告最终为36页、14章、12幅图、9张表。这里提供报告、代码索引及派生仿真结果，不提供原始声呐数据。

## 给网页版AI的阅读顺序

1. 阅读[完整报告正文](海底地形地貌匹配导航算法仿真报告.md)，以当前改进结果为主体，不把历史基线报告当成最新版。
2. 核对[机器可读的完整结果](data/verified_report_results.json)、[主试验独立审查](evidence/main_independent_comparison.json)和[新噪声独立审查](evidence/fresh_independent_comparison.json)。
3. 结合下方算法源码核实数学模型、传感器输入及实现边界；公开文献只能解释方法背景，不能替代本项目实测仿真数值。
4. 如需改写，保留真实数字、试验范围、不利结果和引用，调整研究报告的论述、结构及可读性，不虚构已完成的外场或板载试验。

[下载Word版](海底地形地貌匹配导航算法仿真报告.docx) · [查看PDF版](海底地形地貌匹配导航算法仿真报告.pdf)

## 本轮已完成的结果

| 对照 | 原Global RMSE | 改进Global RMSE | 变化 |
|---|---:|---:|---:|
| 主试验16.48864 km，62458 ping，种子31 | 11.733 m | 5.040 m | 降低57.04% |
| 新噪声3.99986 km，15152 ping，种子33 | 7.629 m | 5.951 m | 降低21.99% |

两组内的底图、初始化和全部导航观测分别保持一致；DR与Local逐值相同，只改后端。主试验横向RMSE由11.519 m降为4.808 m。

改善不是所有指标都提升：主全程终点由7.928 m增至9.940 m；新噪声补充的沿向与10/30 m相对误差略增；用于开发的旧场2 km记录仍有约0.98%的RMSE退化。原始负结果和第一候选均保留在报告与摘要中。新噪声补充不是第二次16.49 km全程，也不是独立地形或独立航次泛化。

## 算法源码

| 文件 | 作用 |
|---|---|
| [SideScanMappedRangeChange.py](../../../standalone/SideScanMappedRangeChange.py) | 两次真实斜距的差分、未知地物高度区间、残差与雅可比 |
| [SideScanMappedRangeGraph.py](../../../standalone/SideScanMappedRangeGraph.py) | 两个可变位姿之间的实际GTSAM非线性因子及信息预算 |
| [SideScanMappedRangeBank.py](../../../standalone/SideScanMappedRangeBank.py) | 因果地图关联、物理ping与通道来源、距离token全局一次性使用 |
| [PureSideScanFeatureNavigation.py](../../../standalone/PureSideScanFeatureNavigation.py) | 既有前端、特征地图、回环、混合后端与新约束的连接 |
| [ReplayPureSideScanNavigation.py](../../../standalone/ReplayPureSideScanNavigation.py) | 同一收到输入的无损重放、真值访问隔离、默认路径保护 |
| [AnalyzeMappedRangeNavigation.py](../../../standalone/AnalyzeMappedRangeNavigation.py) | 完整输出、误差、依赖哈希与首因子前轨迹的独立审查 |
| [AuditMappedRangeInputs.py](../../../standalone/AuditMappedRangeInputs.py) | 距离bin、Local接收器偏移、压力及token来源的独立核对 |
| [BuildMappedRangeNavigationReport.py](../../../standalone/BuildMappedRangeNavigationReport.py) | 本版报告的生成逻辑 |

当前报告使用显式配置 `--mapped-range-changes --mapped-range-policy preserve_legacy`：保留既有地图与回环的信息矩阵，新增距离约束使用自身保守预算。默认关闭新增功能，原海湖算法和原纯侧扫基线没有被自动替换。第一候选 `tempered_legacy` 是对照策略，不是当前推荐配置。

## 改写时必须保留的科学边界

- 当前是使用实测来源外观场生成新虚拟航线回波的受控仿真，不是AUV真实执行该梳形路线的外场绝对定位精度。
- 地图按同航次名义坐标构建，0.3 m为网格间距，不代表独立测绘坐标精度；原始参考和查询记录虽不重叠，仍不构成跨航次独立验证。
- 当前前端为GFTT、SIFT和扫描行几何的纯侧扫适配器，不是官方FAST-LIVO2完整复现。LoFTR用于后端图像对应，新距离因子是本项目独立推导，不冒称完整复现DIASSS或另一论文系统。
- 当前纯侧扫估计器不使用MBES XYZ、海底DEM/高度似然、DVL离底高度、GPS或评价真值。压力深度是接收器垂向状态，不是海底高度。
- 首段没有独立回访时，新距离因子尚不能提供帮助；不通过平底假设、虚构测深或按真实定位误差筛选匹配来消除不可观方向。
- 一次性token和0.5信息预算不等于严格联合相关性建模或统计一致性证明。
- 历史海湖板载验证与当前PC纯侧扫配置分开解释，不能借历史记录宣称新配置已完成板载全程、实时性或全部跨平台精度验收。

## 图表与派生仿真数据

[主全程误差曲线](figures/main_full_errors.png)、[方向误差](figures/main_directional_errors.png)、[实际新增约束](figures/actual_range_constraints.png)、[三维轨迹](figures/improved_3d_trajectories.png)、[新噪声补充曲线](figures/fresh_full_errors.png)均对应保存的完整在线输出。

`data/`保留全部误差CSV、100 m分段统计及无损压缩仿真坐标。这里的轨迹属于虚拟试验，不是原航次GPS轨迹；不包含收到的声呐幅度或原始I/Q。`evidence/`保留数值审查、文档审查、报告生成源码快照和测试摘要。

## 发布范围与复现前提

本次按用户明确要求，将最新代码、报告及报告所含的处理后图表、派生仿真统计发布到Git。原始MAT、实测I/Q、GPS、逐ping声学输入及可还原声学缓存继续不公开。报告PDF、Word与源稿保持已验收字节不变；其中“本地交付”“未上传”和审查JSON中的上传字段描述报告编制阶段，不代表本次发布动作未发生。

网页AI可以读取本目录和源码进行报告改写，但仓库不含本机私有原始数据与全部场景资源，不能据此声称已经重新运行完整声学仿真。报告中的绝对本地路径用于来源追溯，不是在线可下载的数据地址。报告层级保持“题目—一、—1.—（1）”。
