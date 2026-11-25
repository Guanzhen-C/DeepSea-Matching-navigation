# 第七阶段：斜距因子贡献、退化和自适应诊断

**建议继续正式使用 M0。** 完成三个千岛湖历史仿真图的 304 批 × 2 模型的逐批删除，以及三档共 3,431 个原距对的逐项诊断。M1 可在合适高差和互补几何下保留原始距离信息；目前没有外场证据支持替换正式导航，冻结门控仍未启用 M1。

| 问题 | 结论和入口 |
|---|---|
| 逐因子贡献 | 608 次切除尝试，保留 5 次失败。残差大小无法判断帮助方向，航段交互不可相加。[逐对诊断](FACTOR_INFLUENCE.md) · [A–H 消融](COUNTERFACTUAL_ABLATION.md) |
| 哪些物理工况 | 32 条件 × 16 种子扫查；M1 的条件性优势主要在宽差分死区＋合适方向＋原距可用时，近等高继续用 M0。[谱、条件数、误差](GEOMETRY_DEGENERATION.md) |
| Adaptive 为什么没启用 | 代理前缀把可行通过从 2,742 提到 3,353，但 3,431 对仍是窄死区；未保存真实插边协方差，不用真值调阈值。等效短路验证不改变任何门控动作。[门控诊断](ADAPTIVE_GATE_DIAGNOSTICS.md) |
| 标定与跨湖海 | 无独立标靶实测测出 $\rho$；Monterey/Chelan/Jenkinson/Sheboygan 的档案缺可追溯双距来源。[标定](RANGE_NOISE_CALIBRATION.md) · [场景清点](MULTI_SCENE_DATA_AUDIT.md) |
| 论文和下一步 | 贡献是可追溯的诊断与有边界的数理证据，不是已部署的新自适应算法。[论文界限](PAPER_CONTRIBUTIONS.md) · [请补资料](MISSING_MATERIALS.md) |

## 从本机原输入复跑

所有输出使用新目录。冻结图评估与输入选择隔离，只有求解完成后读取真值；逐对 gzip 含原始观测摘要与地理数据，不纳入公开证据。各 JSON 收据记录场景/种子、源输入、配置/程序 SHA-256；[统一清单](evidence/manifest.json)。依赖和定义见[阶段六](../sonar_range_factor_research_v4/README.md)、[阶段五](../sonar_range_factor_research_v3/README.md)与[阶段四](../sonar_range_factor_research_v2/README.md)。

~~~bash
v5_core='output/论文数据资料库_20261124/02_匹配导航算法/复现实验_冻结导航运行时/runtime/core'
v5_policy='docs/reports/sonar_range_factor_research_v4/evidence/policy_frozen.json'
v5_dst=$(mktemp -d)
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$v5_core"
for region in qiandao2 qiandao4 qiandao16; do
  .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.replay \
    --scene "$region" --core "$v5_core" --output "$v5_dst/${region}_loo.json" --workers 3
  .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.diagnostics \
    --scene "$region" --core "$v5_core" --policy "$v5_policy" \
    --detail "$v5_dst/${region}_pair_detail.jsonl.gz" --output "$v5_dst/${region}_diagnostic.json"
  .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.ablations \
    --scene "$region" --core "$v5_core" --diagnostic "$v5_dst/${region}_diagnostic.json" \
    --loo "$v5_dst/${region}_loo.json" --output "$v5_dst/${region}_ablations.json" --workers 3
  .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.influence_summary \
    --detail "$v5_dst/${region}_pair_detail.jsonl.gz" --loo "$v5_dst/${region}_loo.json" \
    --output "$v5_dst/${region}_influence.json"
done
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.geometry_study --output "$v5_dst/geometry_stress.json" --trials 16
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.gate_study \
  --core "$v5_core" --policy "$v5_policy" --output "$v5_dst/gate_study.json"
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.calibration --output "$v5_dst/calibration_selfcheck.json"
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v5.inventory --output "$v5_dst/archive_inventory.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sonar-denoise/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_sonar_range_factor_research_v5.py
~~~

新增 13 项测试与前序 110 项合跑，**123 项全部通过**。本轮仅新研究目录、新报告、新本地输出；没有修改历史数据、正式 M0 或冻结运行时，未运行实时闭环/Jetson，也未向远端上传。
