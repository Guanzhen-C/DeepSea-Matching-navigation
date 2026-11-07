# 海底地形地貌匹配导航

本仓库在OceanSim基础上开展海湖地形匹配、水下声呐惯性导航、嵌入式处理器在环、实测侧扫去噪与纯侧扫特征导航研究。日常开发统一使用 `main`，已验证的历史配置通过标签和固定提交追溯，不再保留多个活动分支。

## 报告与数据入口

- [最新改进算法与网页版AI阅读入口](docs/reports/seabed_matching_improved_algorithm_20261107/README.md)：36页新版报告、源码索引、完整仿真统计及改写边界。16.49 km同输入Global RMSE由11.733 m降至5.040 m，新噪声4 km由7.629 m降至5.951 m。
- [最新报告正文](docs/reports/seabed_matching_improved_algorithm_20261107/海底地形地貌匹配导航算法仿真报告.md) · [Word](docs/reports/seabed_matching_improved_algorithm_20261107/海底地形地貌匹配导航算法仿真报告.docx) · [PDF](docs/reports/seabed_matching_improved_algorithm_20261107/海底地形地貌匹配导航算法仿真报告.pdf)
- [实测侧扫声呐去噪与高清增强完整管线](releases/sonar_denoise_enhanced_20261107/README.md)：原始 MAT → 固定模型 → 边缘修复 → Shotwell 增强，含权重、运行命令、测试与示例图。
- [历史基线仿真报告](docs/reports/seabed_matching_simulation_20261107/海底地形地貌匹配算法仿真报告.md)
- [报告数据与阅读说明](docs/reports/seabed_matching_simulation_20261107/README.md)
- [完整试验指标](docs/reports/seabed_matching_simulation_20261107/data/verified_metrics.json)
- [算法版本索引](config/validated_algorithm_profiles.json)
- [分支整理与历史版本恢复](docs/repository/README.md)

报告包含实际板载湖区47.12 km、海区73.07 km结果，以及实测数据构建场景中的纯侧扫16.49 km完整对照。数据包保留全部误差样本及无损压缩轨迹，提供网页易读的100 m分段统计和来源哈希。原始声呐大文件、个人环境和敏感配置不作为普通Git文件上传。

## 算法与运行

湖区、海区、去噪和纯侧扫分别使用其明确配置，不将不同试验的观测条件混用。当前纯侧扫改进通过 `standalone/ReplayPureSideScanNavigation.py` 或 `standalone/RunPureSideScanFeatureShort.py` 显式开启 `--mapped-range-changes --mapped-range-policy preserve_legacy`；不带开关时不启用新增距离约束。`standalone/RunValidatedSideScanNavigation.py` 保留为此前四组完整验证的入口，不应误称它已默认切换到本轮改进。海湖板载固定源码打包入口为 `standalone/hil/prepare_bundle.py`。完整仿真重新执行还需本机私有场景和观测资源，报告数据包并非原始声学资源镜像。

硬件在环是模拟传感器输入、实际目标处理器运行的验证，不是实船外场实验。海区跨架构精度及持续实时性尚未全部通过；纯侧扫结果也不是独立实测地理精度验收。具体成功、失败和适用边界见报告。

## 原项目

原始框架来自 [umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim)，原项目介绍、安装和场景说明见 [docs/README.md](docs/README.md)。原项目署名、引用及许可证保留；本仓库新增算法的验证结论不代表原项目官方承诺。
