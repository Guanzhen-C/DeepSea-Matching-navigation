# 复跑和后续验证

只需发布阶段文档、研究代码和聚合收据；原始声呐、轨迹、底图和仿真真值均留在既有档案。以下从仓库根目录执行，输出用新文件。PC 不需要 Isaac Sim GUI；固定图回放需要历史本机数据。

~~~bash
audit_core='output/论文数据资料库_20261124/02_匹配导航算法/复现实验_冻结导航运行时/runtime/core'
audit_out=$(mktemp -d)
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v4.adaptive \
  --output "$audit_out/policy_frozen.json"
.venv-sonar-denoise/bin/python -m research.sonar_range_factor_v4.benchmark \
  --output "$audit_out/batch_benchmark.json"
for audit_scene in qiandao4 qiandao2 qiandao16; do
  PYTHONPATH="$audit_core" .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v4.replay \
    --scene "$audit_scene" --frozen-core "$audit_core" \
    --policy "$audit_out/policy_frozen.json" --output "$audit_out/$audit_scene.json"
done
PYTHONPATH="$audit_core" .venv-sonar-denoise/bin/python -m research.sonar_range_factor_v4.validation \
  --frozen-core "$audit_core" --policy "$audit_out/policy_frozen.json" \
  --output "$audit_out/validation.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$audit_core" \
  .venv-sonar-denoise/bin/python -m pytest -q -p no:cacheprovider tests/test_sonar_range_factor_research_v4.py
~~~

已生成收据来自本机实测，不是硬实时验收。Eigen 探针独立编译/执行，不是 Python 实验依赖。本轮 10 项独立测试与前序 100 项回归合跑，共 110 项通过。之后应补真实原距协方差、独立关联概率、因果插边姿态与预测不确定度，恢复 Monterey/Chelan 的原距来源链，再冻结新策略和多场景留出比较；不可用本轮结果反向挑阈值。C++ 路径下一步移植完整箱枚举与载荷批处理，按同一 Decimal 证据和退化集回归。
