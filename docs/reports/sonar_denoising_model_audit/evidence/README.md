# 可随审查发布的证据

- [run_receipts.json](run_receipts.json)：从本机归档使用字段白名单提取。工件 SHA-256 从本机二进制重新计算；运行计时从相应 JSONL 字段重算均值、P50/P95/P99、最大值和超过 200 ms 样本数。原回执和轨迹文件只公布文件名及摘要，不附轨迹、坐标、I/Q、数组或路径信息。
- [SOURCE_EXCERPTS.md](SOURCE_EXCERPTS.md)：带原始行号的装配、训练、特征构造和实际推理摘录；不假装是完整软件包。H0 归档的实验入口源码 SHA-256 与摘录所用本机原文件一致。
- [navigation_aggregates.csv](navigation_aggregates.csv)：归档指标表的 24 条汇总数据，含 DR、Local、Global 的 RMSE、P95、最大和终点误差，不含轨迹坐标。两个仿真批次的三维真值与实采同航次二维 GNSS 参考偏差不能混在同一精度口径。

核对顺序：

| 需要确认 | 公开证据所对应的 JSON 字段 |
|---|---|
| 三件训练权重是哪几件 | `wavelet_training.selected.SHA256`、`spectral_training.selected.SHA256`、`protection_selection.selected.SHA256` 分别等于 `artifacts` 中的 `torch_wavelet`、`torch_spectral`、`torch_protection`；账本原名与冻结中转名可以不同。 |
| 确实在板端使用，而非 PC 文件名推测 | `H0.model_SHA256` 和 `H5.engine_sha256` 都等于 `plan_R12721`；`real_track.engine_sha256` 等于 `plan_R12189`；H0 记录全部 62,458 ping 非零修改。 |
| 32 个实采 ping 与数值对照区别 | `recorded_pilot.pings=32`，`recorded_pilot.paired_gain_checks` 只有原来的序号 0、1、2；`pc_preflight.actual_pings_checked_against_onnx=3` 属于独立 PC→ONNX 核对。 |
| 5 Hz 实时边界 | `H5.simulated_hz=5`，`sustained_wall_5Hz_certified=false`；`latency_ms.model_ms` 与 `latency_ms.denoise_ms` 是不同分段。原始计时位置见源码摘录及[性能说明](../PERFORMANCE.md)。 |
| 为什么不是 56M | 三个实际 `.pt` 摘要与 56M 的推理权重不同，且 `wavelet_training.offline_teacher_training_weight=0` 只说明选定骨干阶段，不推断全部前代研究。 |

摘要用于对齐记录和本机原文件，**不是**第三方获得原件的下载地址；产生同一哈希的文件不因网页显示这份摘要就已上传。还缺目标板完整 TensorRT 构建日志与同一次运行的资源明细。
