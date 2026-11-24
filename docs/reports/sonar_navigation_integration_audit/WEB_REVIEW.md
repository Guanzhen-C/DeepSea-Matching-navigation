# 网页端论文审阅入口：侧扫去噪—导航的真实接口

本页和所有下列材料均在 GitHub，供无法读取研发机 `output/` 的审阅者使用。这是第二阶段，第一阶段[模型身份、权重指纹与性能](../sonar_denoising_model_audit/WEB_REVIEW.md)仍是模型本身的证据入口。

## 阅读顺序

1. [第二阶段总览](README.md) · [实际板端流程与运行记录](END_TO_END_FLOW.md)
2. [去噪输出接口](DENOISING_INTERFACE.md) · [导航输入、测距与因子图](NAVIGATION_INTERFACE.md) · [原行号源码摘录](evidence/SOURCE_EXCERPTS.md)
3. [数据一致性](DATA_CONSISTENCY.md) · [脱敏配对回执](evidence/paired_run_receipts.json) · [定位汇总 CSV](../sonar_denoising_model_audit/evidence/navigation_aggregates.csv) · [特征数量与风险](POTENTIAL_ISSUES.md)
4. [拟开展的 2×2 四组实验](FOUR_ARM_EXPERIMENT.md) · [论文贡献及表述边界](PAPER_CONTRIBUTIONS.md) · [待修订的中文论文原稿](../qiandao_sidescan_manuscript_20261124/manuscript_zh.pdf)

## 可交给网页端的审阅要求

请按上述证据修改中文论文的接口、导航几何和实验章节。不把计划写成结果，不用本机路径推断网页看不到的信息。重点保留：

- **实采幅度和仿真复 I/Q 两条路径不同。** 实采对数增益乘到接收幅度后直接构包，不重新脉压；仿真在脉压前恢复复 I/Q，旁路/恢复两臂分别通过同一个接收器脉压后构包。
- 去噪不重排原生样点，但下游每束汇聚成 512 个导航距离格；子图还有外观插值和来源选择。区分物理刻度、格中心与原生逐点分辨率，不能把图像坐标或模型增益写成 GTSAM 的量测。
- 同输入配对已经在 NX 跑通：62,458 ping 与 103,055 ping 的两次仿真，以及 2,020 个实际接收、1,010 个查询的同航次回放。仿真和实采评分口径不同；现有对照只覆盖 C/D，原始导航的 A/B 和交互作用尚未运行。
- 短 RootSIFT 小试确实少了关键点和内点，但与正式前端实现不同；正式前端和地图成功次数按批次呈不同方向。没有证据确认少了哪一舷的有效弱地物，也没有独立弱特征标签。
- 区分已运行的完整板端链路、归档可审计性、全新环境的独立复跑和持续实时验收。不要把前两项写成后两项，也不要把同航次 GNSS 偏差写成独立水下定位真值。

公开内容仅为审查文档、引用源码和脱敏汇总，不包含原始回波、原始轨迹、GNSS、地图栅格、模型权重或计划二进制。
