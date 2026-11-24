# 第五阶段：M1 高度剖面改进与导航验证准备

**本轮不替换 M0。** M1 已具备有明确适用域的严格高度目标值证书、相关噪声整块白化、鲁棒幅值代价及独立 GTSAM 接口；它在大垂向高差的受控条件下有优势，但这次固定关联历史**仿真**图重放总体 RMSE 略差于 M0。M2 仅作多视图扩展。

- [严格求解与保证边界](M1_GLOBAL_SOLVER.md) · [失败案例](FAILURE_CASES.md)
- [鲁棒原距统计模型](M1_ROBUST_ESTIMATION.md) · [公平对照协议](FAIR_COMPARISON.md)
- [M2 可观性和图代价](M2_MULTI_VIEW.md) · [独立 GTSAM 接口和回退](GTSAM_INTEGRATION.md)
- [实际合成试验、固定图重放和一致性结果](MONTE_CARLO_RESULTS.md) · [后续实验](EXPERIMENT_PLAN.md) · [论文表述及选型](PAPER_CONTRIBUTIONS.md)

证据分级：数学上的目标界、驻点覆盖和同白化 Schur 等价是**条件证明**；单测、合成、同源仿真图重放是**已执行的数值验证**；完整实时闭环、固定滞后和独立外场是**待验证**。没有修改正式 M0、已有历史档案或第四阶段结果。新实现位于 [research/sonar_range_factor_v3](../../../research/sonar_range_factor_v3)，[独立测试](../../../tests/test_sonar_range_factor_research_v3.py)与[第四阶段基线](../sonar_range_factor_research_v2/README.md)可对照。
