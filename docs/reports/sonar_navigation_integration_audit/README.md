# 第二阶段代码审查：7.52M 去噪如何进入纯侧扫导航

审查基线：`DeepSea-Matching-navigation/main@b461dbb`；接续[第一阶段模型审查](../sonar_denoising_model_audit/README.md)。本次仅新增审查文档。读冻结导航源码和本机日志、核对摘要、重算配对计数；未修改原算法、权重和已有实验数据，未重跑 Jetson。

**只看 GitHub：**从 [网页版审阅入口](WEB_REVIEW.md) 开始；[脱敏配对回执及源码摘录](evidence/README.md) 不依赖审阅者访问本机目录。

## 结论

**已有真实、完整、已在 Jetson Orin NX 跑通的去噪—导航链路；在原归档与资产齐备条件下可按记录复核和准备复跑，但尚未验证仅凭 GitHub 从干净环境独立复跑，也未验证持续 5 Hz 实时。**

- **实采重放（R）**：真实接收复 I/Q → 模型输出 log gain → 按预存支持恢复**幅度** → `received_packet()` → `PureSideScanFeatureNavigator.side_scan()` → DR／Local／Global。2,020 个物理 ping 均进模型，只有奇数索引的 1,010 个 ping 进入两导航臂。已完成同航次回放，定位评价是二维 GNSS **参考偏差**，不是独立实海真值。
- **仿真 HIL（H0、H5）**：共同的虚拟接收复 I/Q → 一臂旁路、另一臂恢复**复 I/Q** → 两臂分别执行**相同的一次**脉冲压缩 → 相同接收包转换及冻结导航。H0 完成 62,458 ping；H5 完成 103,055 ping。这不是只生成图像或先离线去噪再回放处理结果。
- **索引边界**：模型恢复不重排 2×8 路或原生距离格；导航接收包会把能量汇聚到每束 512 格，子图还会插值/投影。物理斜距刻度保留，但 GTSAM 中的距离观测是**导航格中心及其来源**，不是完整原生样点索引。
- **特征数量下降属实，但不是统一结论**：32-ping RootSIFT 小试中查询关键点 2,160→1,428、比值通过匹配 83→21、RANSAC 内点 17→8。该试验不是实际导航的 GFTT/SIFT 前端，也不是 LoFTR 后端。H0 和 H5 导航前端通过次数均下降，R 实采前端通过次数反而上升；全程 Global RMSE 与特征/内点数量不能画等号。尚无独立弱纹理真值来区分删掉的是伪条纹特征还是有效地物。
- **四组实验尚未完成**：现有严格对照是同一导航版本下的旁路/去噪，可作为 C/D 证据；历史“不额外加条纹”不是 A/B，也不能比较“原始导航”和“改进导航”的因果效果。

## 证据约定与索引

**已运行**：存在设备回执和完成记录；**已验证**：本审查通过冻结源码、摘要或逐行日志重算核对，限所列范围；**仅有代码**：实现存在，不表示相应功能在某工况实际触发；**待实验**：尚无对应独立实测。R/H0/H5 是三次不同的运行，不能跨批混成一组对照。

| 简称 | 原始证据位置（以仓库根目录为起点） |
|---|---|
| N | `output/论文数据资料库_20261124/02_匹配导航算法/` |
| S | `output/论文数据资料库_20261124/01_声呐去噪模型/` |
| F | `N/复现实验_冻结导航运行时/runtime/`；导航源码取此版本；部署 manifest 在其上一级，不取工作区同名文件覆盖 |
| R | `N/实采1_40km_2020ping_同航次重放/{run_summary.json,trajectory.jsonl,actual_recorded_route_noisy_AB_audited_v2.json}`；同目录 `同航次先验图与GNSS索引/prior/metadata.json` |
| H0 | `N/完整梳形_62458ping_主对照/NX本次全程/{run_definition.json,run_summary.json,trajectory.jsonl}` |
| H5 | `N/完整梳形_5Hz_103055ping_补充对照/带条纹双臂/{run_definition.json,run_summary.json,trajectory.jsonl}` |
| P32 | `S/7_52M_NX实采推理核对/wired_ab_pilot.json`；只有前 3 ping 回传增益做 PC–NX 逐样点对比，32 ping 的图像小试不是导航真值试验 |

定位代码引用凡写 `F/standalone/...` 均指冻结导航源；`standalone/...` 指对应的本机实验脚本/接收器。归档路径含受限数据，只在本机复核。[第一阶段公开回执](../sonar_denoising_model_audit/evidence/run_receipts.json)与[指标汇总](../sonar_denoising_model_audit/evidence/navigation_aggregates.csv)可不访问这些原始文件。

## 分卷

- [去噪输出接口](DENOISING_INTERFACE.md) · [导航输入与因子图](NAVIGATION_INTERFACE.md) · [实际端到端流程](END_TO_END_FLOW.md)
- [四组实验](FOUR_ARM_EXPERIMENT.md) · [数据一致性](DATA_CONSISTENCY.md) · [技术风险与复核指标](POTENTIAL_ISSUES.md)
- [论文贡献与可发表边界](PAPER_CONTRIBUTIONS.md)
