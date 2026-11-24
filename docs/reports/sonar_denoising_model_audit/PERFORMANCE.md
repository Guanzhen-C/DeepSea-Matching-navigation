# NX 性能证据和实时性边界

证据位置和制品摘要见 [README](README.md)。复算只读已存日志，未重启 NX、未做新推理。记 P95、P99 为样本经验分位（NumPy 线性插值），均值/最大取原样本，不裁长尾。

## 1. 硬件与软件

| 项目 | 记录与置信范围 |
|---|---|
| 板卡 | B 的 `hardware_verified.device_tree_model`：`NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super`；CUDA 可用物理总量快照 16,416,002,048 B，8 个 SM，计算能力 8.7；架构 aarch64。R/H5/H0 的目标证明另见模型及板端脚本回执。 |
| L4T / JetPack | B 有 **L4T 36.4.3** 原记录，其标准映射为 **JetPack 6.2**，参见 [NVIDIA Jetson Linux 36.4.3](https://developer.nvidia.com/embedded/jetson-linux-r3643)；没有每次小模型运行的 `nvidia-jetpack` 软件包快照。 |
| CUDA / TensorRT / PyTorch | B：CUDA runtime 12.6、TensorRT 10.3.0、cuDNN 90300、PyTorch `2.5.0a0+872d972e41.nv24.08`；H5/R 的 LoFTR/板端回执中也可核对 PyTorch。该组合来自同板历史环境和运行子系统记录，构建制品仍须补单独环境锁。 |
| 功率和线程 | B 曾观测 40 W mode 4；不能推定每次小模型运行始终如此。H0/H5/P32 的板端处理代码明确 PyTorch intra-op 2、OpenCV 1（`RunQiandaoHistoricalClockAB.py:105–116`、`RunQiandaoNoisyCombBoardAB.py:55–65`、`RunQiandaoRecordedSmallBoardPilot.py:214–217`）；H0/H5 宿主 PyTorch 1。没有该两次运行的核亲和、实际峰值线程数或热节流时序。 |

## 2. 7.52M 档案实测：各阶段分开统计

以下各行都是对应日志的实际样本。单位 ms；“>200”只说明该**分段**测量超过 5 Hz 一个周期，不等同于整链丢包率。

| 批次与日志字段 | N | 平均 | P50 | P95 | P99 | 最大 | >200 样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 历史时钟梳形，`model_ms` | 62,458 | 125.57 | 125.58 | 130.02 | 132.85 | 140.54 | 0 |
| H5 5 Hz 梳形，`model_ms` | 103,055 | 126.98 | 126.51 | 129.31 | 130.78 | 262.04 | 371 |
| H5，`denoise_ms` | 103,055 | 227.98 | 227.69 | 237.73 | 240.00 | 421.96 | 102,448 |
| R 实采同航次，`model_ms` | 2,020 | 130.84 | 104.74 | 221.83 | 231.59 | 236.80 | 306 |
| R，`full_denoise_ms` | 2,020 | 187.75 | 156.91 | 308.25 | 330.54 | 387.30 | 629 |
| P32 实采小试，`model_ms` | 32 | 113.23 | 114.14 | 116.09 | 123.68 | 126.99 | 0 |
| P32，`full_denoise_ms` | 32 | 178.98 | 168.38 | 197.10 | 402.49 | 492.84 | 2 |
| P32，`host_ack_roundtrip_ms` | 32 | 354.51 | 337.27 | 434.87 | 640.30 | 730.02 | 32 |

- `model_ms`：`NativeLearnedGain.run()` 中，完成 I/Q→特征之后，CUDA Event 计 `Runtime.run()` 的绑定数据复制、图回放及图内状态更新。未计 CUDA 前置 I/Q 特征计算、后续 CPU 取值、恢复、导航、网络。
- H5 `denoise_ms`：`RunQiandaoNoisyCombBoardAB.board:96–107`，包含推导水柱掩码、特征、板端推理、取回、乘回到 I/Q、校验；不含后面的脉压和双导航。
- P32/R `full_denoise_ms`：`RunQiandaoRecordedSmallBoardPilot.serve:247–276`，止于 `stream.run(iq)` 返回；**包含板端特征和模型，但计时结束后才恢复幅度**，也不含导航网络。字段名不能与 H5 `denoise_ms` 直接等同。R 的 `navigation_ms` 在偶编号非查询 ping 为零，查询还运行两臂；不将其误报为单臂导航耗时。
- `host_ack_roundtrip_ms` 是 P32 宿主发起至回执全过程，前 3 个还有校验用增益回传。它不是纯网线 ping，也不能作为单臂常态测试的替代。

| 完整流程 | 测量结果 | 限定 |
|---|---|---|
| H0 全梳形 | 47,705.10 s / 62,458 ping，1.309 ping/s；仿真间隔约 0.330 s | A/B 双臂墙钟总时，不是模型离线推理总耗时 |
| H5 全梳形 | 80,053.46 s / 103,055 ping，1.287 ping/s | 仿真时间轴 5 Hz，日志 `sustained_wall_5Hz_certified=false` |
| R 实采整段 | 1,000.51 s / 2,020 ping，约 2.019 ping/s | 原始时间戳重放、1,010 个偶奇分开后的查询；两臂、含接收与导航，不是持续设备采样验收 |
| 小模型启动分段 | **尚无测量** | 有 deserialize / create context / warm-up / capture 源码，未找到各阶段耗时；不能把首 ping 或整段计时减成模型加载时间 |
| 小模型 GPU/CPU 峰值、GPU 利用率 | **缺同运行证据** | 未找到按 J 引擎摘要绑定的全程 tegrastats、进程 VmHWM 和 TRT 上下文资源记录；CUDA RAM 总量不是峰值 |

历史 `N/完整梳形_62458ping_主对照/历史基线板端回传/tegrastats.log`（无新增条纹历史基线）重计 20,599 行：共享 RAM 最大 5,850 MB，GR3D_FREQ 峰值 99%、等样本平均 1.97%；`output/nx_super_hil_20261110/sea_r2_completed_board_review_v1/tegrastats.log` 5,723 行：5,286 MB、79%、3.76%。这些是历史导航系统资源，不能拿来填实际 7.52M 的独占显存或 GPU 利用率。H 中 `NX_FP32_and_graph_v1`、`NX_actual_native_precision_v1` 是不同协议/候选的探针；不得把其中 `core200` 等子窗数据作为本模型原生全宽实测。Git 中 `demo/hil/jetson_20261021`、[三场景报告](../jetson_hil_research_bundle_20261112) 给导航链背景，不是 W/J 身份来源。

## 3. 56M 在同板也跑过，但不是实时逐 ping

F56 的原始 NX TensorRT 全幅报告：2,020×16×12,189 数值，核心 192×768，**4018.34 s**（报告 `elapsed_s`，一次离线全段）；严格全局 FP32 数值检查的最大增益差约 $2.575×10^{-5}$。`TensorRT_context_bytes=1,080,465,408`，观测全局 GPU 空闲变动最大 799,920,128 B、Torch allocator 峰值 12,923,904 B、进程峰 RSS 1,875,268 KiB。它们是不同记账范围，不能相加或当作全系统 GPU 峰值。另一次旧 NX **PyTorch** 96×768 全段：4171.44 s，CUDA 推理累计 4061.16 s（tile 中位 0.79446 s），Torch CUDA 分配峰值 2,871,822,848 B。不要把上述耗时平均后称为可流式传感器延迟。

## 4. 实时性可发表口径

**已验证**算法可对每个原生 ping 顺序处理并持久化状态。**尚未验证**持续 5 Hz 实时去噪/导航。5 Hz 仿真输入周期是 200 ms；H5 模型 P99 有裕量，但完整板端去噪段在该批中 102,448/103,055（约 99.41%）超过 200 ms。单臂、直接板载输入是否可借降低预处理成本、异步流水、避免往返等达到 5 Hz，需要对同一小模型另外运行，并报告吞吐、端到端 P99、超时率、丢包和功耗热稳态。真实设备运行速率还应由实测 ping 间隔和配置确认；P32 原实采间隔约 0.33 s，不代表所有声呐工况。
