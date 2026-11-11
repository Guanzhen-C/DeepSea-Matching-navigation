# 三场景Jetson硬件在环仿真报告与研究资料

本资料包用于重新撰写科学严谨的海底地形地貌匹配导航研究报告。包含三组实际Jetson执行导航的完整轨迹、逐样本位置误差、技术报告、图表、实验配置和核验记录。千岛湖采用Orin NX Super结果；Monterey和Chelan采用历史AGX Orin完整结果，不以未完成的新NX测试替代。

优先阅读：[网页版研究入口](WEB_REVIEW.md)、[完整报告正文](REPORT.md)、[数据字典](DATA_DICTIONARY.md)、[全采样复核指标](data/verified_metrics.json)。报告正文是待审阅材料，不是免于质疑的最终科学结论。

## 一、报告文件

- [PDF排版报告，34页](海底地形地貌匹配导航算法仿真与Jetson硬件在环验证报告.pdf)
- [可编辑Word报告](海底地形地貌匹配导航算法仿真与Jetson硬件在环验证报告.docx)
- [便于网页读取的完整Markdown正文](REPORT.md)
- [11幅报告插图](figures)

本包完整结果的统计对象是实际板端在线位置输出，不是上位机参考算法输出，也不是事后对齐、平滑后的替代轨迹。所有以下指标基于完整样本，不基于抽样预览。

## 二、三组完整实验

| 场景 | 实际导航设备 | 最后样本累计航程 | 同步评价样本数 | Global三维RMSE | Global P95 | Global最大误差 | Global末点误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| Monterey海底 | Jetson AGX Orin | 73.070329 km | 243569 | 6.830 m | 13.267 m | 41.453 m | 4.451 m |
| Chelan湖底 | Jetson AGX Orin | 47.116489 km | 157056 | 6.829 m | 13.527 m | 38.784 m | 22.517 m |
| 千岛湖实测来源场景 | Jetson Orin NX Super 16 GB | 16.488644 km | 62458 | 4.871 m | 10.462 m | 19.315 m | 5.673 m |

这里的海底、湖底实验均为实测来源环境中的导航处理器在环仿真，不是真实AUV外场航行。三组地图、算法分支、输入和航线不同，不构成硬件性能或算法精度的公平排名。

## 三、完整数据和核验材料

| 文件 | 内容 | 是否全量 |
|---|---|---|
| [sea_hil_trajectories_full.csv.gz](data/sea_hil_trajectories_full.csv.gz) | 海底真值、DR/Local/Global位置、误差及部分时序诊断 | 243569行完整记录 |
| [lake_hil_trajectories_full.csv.gz](data/lake_hil_trajectories_full.csv.gz) | Chelan同类轨迹及误差 | 157056行完整记录 |
| [qiandao_NX_navigation_states.npz](data/qiandao_NX_navigation_states.npz) | NX千岛湖时间、航程、四通道位置及三通道误差 | 62458行完整记录 |
| [sea_complete_error_samples.csv.gz](data/sea_complete_error_samples.csv.gz) | 海底全样本三维误差 | 完整 |
| [lake_complete_error_samples.csv.gz](data/lake_complete_error_samples.csv.gz) | Chelan全样本三维误差 | 完整 |
| [qiandao_complete_error_samples.csv.gz](data/qiandao_complete_error_samples.csv.gz) | 千岛湖全样本三维误差 | 完整 |
| data/*_error_preview_only.csv | 便于网页查看的等索引抽样 | 每场景最多2000行，不用于统计或寻找峰值 |
| [evidence](evidence) | 完整性、相对误差、资源、部署与输入指纹 | 选择性导出并脱敏 |
| [NX实际部署算法源码](algorithm_snapshot/qiandao_NX) | 与部署清单逐文件SHA256一致的Python导航算法 | 算法审阅快照，非完整可运行分发包 |

下载完整仓库后，在本目录执行：

```bash
python -m pip install numpy
python verify_bundle.py
```

该脚本检查资料包文件哈希，从完整轨迹重算三维位置误差，并核对完整误差CSV、RMSE、P95、最大值及末点值。无需Jetson、原始MAT、仿真器或模型权重，不重新运行导航估计。

## 四、必须保留的科学边界

1. 海底全程定位输出完整，但原部署的资源与相对精度检查未全部通过：上位机源进程出现OOM，且部分指标相对参考结果退化。完整结果可用于分析，不等于整套部署验收通过；原始传感器归档仍有完整性核验待完成。
2. Chelan通过已有完整性和相对精度检查，但不满足严格硬实时验收条件。千岛湖也不能仅凭接近1倍的端到端速度证明硬实时。
3. 千岛湖参考场与查询场来自同一航次的不同物理记录，坐标布局采用名义航迹，不是独立高精度地理配准或跨航次外场定位验证。
4. 千岛湖是纯侧扫特征导航：未向导航器输入MBES或DVL离底高度；重复观测的斜距变化用于约束，未知目标高度作为待消元量，不能写成使用真实离底高度定位。
5. Local仍有长程漂移。千岛湖Local RMSE略低于DR，但P95和末点误差略高。不得仅以Global表现较好证明前端长期无漂移。
6. 初始化为受控条件；没有完成未知初始位置的全地图冷启动重定位验证。
7. NX全16路模型去噪在场景生成之前离线完成，不是已经验证原生全16路声呐实时5 Hz处理。

## 五、公开范围与数据保护

已公开：派生导航轨迹、位置误差、统计、报告图、配置和脱敏审查信息，以及用于算法审阅的冻结源码。数值不因脱敏而调整。报告的Word、PDF及Markdown与本地定稿逐字节一致。

未公开：声学所原始MAT、逐ping声呐幅度阵列、实测复数信号、原生完整侧扫图、参考/查询强度场数组、模型权重、TensorRT引擎、网络凭据。声呐插图仅是明确命名的等比例preview_only预览，不能据此恢复原生测量分辨率。原生正式图片仍为24378×8084，未拉伸或覆盖。

资料包用于结果复核和报告重写，不包含重新执行三套导航所需的全部原始输入与依赖。完整派生结果、原始传感器归档完整性和部署验收状态必须分别表述。
