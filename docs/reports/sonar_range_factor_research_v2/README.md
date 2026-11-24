# 第四阶段：侧扫原始斜距模型改进与对照验证

**结论：保留 M0；将 M1 作为论文的主候选，M2 作为多视角扩展。** 本阶段已完成推导、独立原型、合成和小图 GTSAM 验证；未改正式导航、未改冻结数据，未以合成结果冒充实采、HIL 或闭环验证。

| 必答问题 | 审查结论 | 证据 |
|---|---|---|
| 1. M0 是否值得保留？ | 值得。最简单、已验证、低成本；对照中也有优于新模型的工况。 | [M0](MODEL_M0_REVIEW.md) · [数值](MONTE_CARLO_RESULTS.md) |
| 2. M1 是否更符合原始物理？ | 是。在下视直达模型、SPD 权重下，剖面零代价当且仅当同一个高度解释全部原距；M0 不具备该等价。非等高通常还保留局部方向，但不能承诺所有几何下提升。 | [M1](MODEL_M1_PROFILED_RANGE.md) · [可行性](PHYSICAL_FEASIBILITY.md) |
| 3. M2 是否增加有效信息？ | 有条件增加。正确身份、共享协方差和几何有效时，同一批原距的一阶信息可严格增加；不是多画图边就有增益，错配时可能更差。 | [M2](MODEL_M2_SHARED_LANDMARK.md) · [可观性](OBSERVABILITY_COMPARISON.md) |
| 4. 哪个最适合论文主方法？ | M1 优先作主候选：物理修正明确，不新增长期状态，有可检验的导数和双位姿接口。不等于已具备替换生产的资格。 | [优化](JACOBIAN_AND_OPTIMIZATION.md) · [准入](EXPERIMENT_PLAN.md) |
| 5. 有独立理论价值吗？ | 有潜力，尚不能宣布。贡献只能归于该观测设置、边界/退化刻画与因子设计，而非已知原距几何/联合优化/消元本身。 | [论文范围和先例](PAPER_CONTRIBUTIONS.md) |
| 6. 投稿前还缺什么？ | 标靶和真实相位中心/声速/时钟/深度标定，独立地图和位置真值、已核实反射点身份，以及固定图消融、闭环、多航次置信覆盖和时延。 | [统计](STATISTICAL_CONSISTENCY.md) · [实验](EXPERIMENT_PLAN.md) |

研发依据：[第三阶段斜距理论审查](../sonar_navigation_theory_audit/README.md)、[第二阶段接口审查](../sonar_navigation_integration_audit/README.md)、[第一阶段去噪模型审查](../sonar_denoising_model_audit/README.md)。源码和测试都独立：[研究模型](../../../research/sonar_range_factor_v2/models.py)、[GTSAM 适配器](../../../research/sonar_range_factor_v2/gtsam_adapter.py)、[合成执行器](../../../research/sonar_range_factor_v2/experiment.py)、[单测](../../../tests/test_sonar_range_factor_research_v2.py)。合成证据不含原始实采数据。
