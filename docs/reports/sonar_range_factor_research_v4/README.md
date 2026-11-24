# 第六阶段：M1 批量加速、Adaptive 和多场景固定图验证

**正式导航继续用 M0。** 同一节点对的 48 个地物单次冷启动剖面由 18.22 ms 降至 1.95 ms，约 9.35 倍；整套小图求解只从 74.37 ms 降至 35.49 ms，约 2.10 倍，未达到整图十倍目标。三份千岛湖历史仿真因子流已复跑。当前 Adaptive 在三个档案的初始化条件下全部保留 M0，不是一次成功的自适应精度提升验证。

| 材料 | 内容 |
|---|---|
| [批量计算与可核验精度](BATCH_OPTIMIZATION.md) | 预计算、热启动、驻点筛选、雅可比/Hessian、GTSAM 批因子与 Eigen 探针 |
| [Adaptive 设计和门限](ADAPTIVE_RANGE_FACTOR.md) | 因果输入、有界检查、回退及信息权重，门限冻结证据 |
| [五场景档案及实测结果](FROZEN_GRAPH_VALIDATION.md) | 千岛湖 16.49/4/2 km 的固定关联回放；Monterey/Chelan 缺出处 |
| [PC 在线准备](PC_INTEGRATION.md) | 小图 iSAM2 插入/撤销/恢复、固定滞后越窗安全；未启动闭环 |
| [复跑和下一步](REPRODUCIBILITY.md) | 证据位置、命令、验收要求和限制 |

阶段五的 [M0/M1/M2 结论](../sonar_range_factor_research_v3/README.md)仍适用。本轮所有代码在 [独立研究目录](../../../research/sonar_range_factor_v4)，没有改正式算法、历史数据、前序档案或冻结运行时。M1_Gaussian/M1_Robust 留作研究，M2 仍为扩展。
