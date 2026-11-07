# main 分支与已验证算法索引

## 一、分支整理结果

2026年11月7日，将已验证纯侧扫仿真基线 `f3476fb170db86570a897587562b57075d135d6e` 快进整合到 `main`。整理前全部本地分支及项目远端分支的提交均为该提交的祖先，没有重写算法或丢弃未合并提交。

本地和 `Guanzhen-C/DeepSea-Matching-navigation` 只保留 `main` 分支。原18个其他本地分支、15个其他项目远端分支已删除引用；原有工作目录内容没有删除。官方上游 `umfieldrobotics/OceanSim` 未修改。

## 二、恢复与追溯

删除分支前，已创建并验证完整 Git bundle：本机 `.git/main-consolidation-20261107.bundle`。原35个本地及项目远端引用分别保存为 `archive-20261107/local/…`、`archive-20261107/remote-deepsea/…` 标签，并核对GitHub端提交一致后才删除远端分支。

逐引用提交和归档标签见 [main-consolidation-20261107.json](main-consolidation-20261107.json)。恢复某个开发分支时可从对应标签新建分支；日常工作仅在 `main` 继续。不要把归档标签误认为当前算法仍分散在多个活动分支。

## 三、四类算法入口

| 用途 | 固定标签 | 主要代码与配置 |
|---|---|---|
| 湖区地形匹配基线 | `lake-sim-baseline-20261021` | `standalone/run_chelan_guarded.py`、`standalone/hil/prepare_bundle.py` 的 lake 固定源 |
| 海区地形匹配基线 | `sea-sim-baseline-20261021` | `standalone/run_monterey_72km_stable.sh`、`standalone/hil/prepare_bundle.py` 的 sea 固定源 |
| 固定实测声呐去噪 | `sonar-denoise-frozen-20261106` | `releases/sonar_denoise_20261106/`、`standalone/FreezeSonarDenoisingRelease.py` |
| 纯侧扫特征导航仿真 | `pure-sss-sim-validated-20261107` | `standalone/RunValidatedSideScanNavigation.py`、`standalone/SideScan*.py`、`standalone/PureSideScan*.py` |

这四类用途的代码和版本历史已统一纳入 `main`。精确复现旧湖区、海区验证时，仍按固定提交提取执行源码，而不是把当前树中的任意参数组合称为旧基线。硬件在环打包器按固定提交提取源码，删除分支名称不影响提交或提取能力。

机器可读索引见 [validated_algorithm_profiles.json](../../config/validated_algorithm_profiles.json)。原始历史记录中的 `baselines/lake-chelan` 等分支名称是当时的版本标识，现应使用此处标签或提交，不需要恢复同名分支。

## 四、结果解释

湖区和海区基线指真实地形上的仿真及处理器在环实验，不是已经完成实船湖试或海试。板载全程输入输出已保留，但海区跨架构精度规则和持续实时性没有全部通过。千岛湖纯侧扫方案在实测数据构建的场景中获得完整对照收益，不等于独立实测地理精度验收。

敏感配置、原始声呐大文件、个人环境和其他无关未跟踪文件没有因分支整理而加入Git。报告需要的数据会使用有来源哈希的轨迹、误差、分段统计和图表整理，不将大体积原始采集包误称为已经上传。
