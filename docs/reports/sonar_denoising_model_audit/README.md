# 侧扫声呐去噪模型专项审查

审查基线：[DeepSea-Matching-navigation/main @ eb5021a](https://github.com/Guanzhen-C/DeepSea-Matching-navigation/tree/eb5021a1874c89e1c5bd0a12b592bbf90445313d)。范围为该 Git 树（含已删除文件的基线内容）、本机工作区及本机运行归档。读取记录、重算日志统计、校验制品摘要、在 CPU 严格加载权重并计数；未重跑实验、未连接板端重测、未改算法和数据。

**只看 GitHub：**从 [网页版审阅入口](WEB_REVIEW.md) 开始。部分关键证据已整理成 [脱敏运行回执及制品指纹](evidence/run_receipts.json) 与 [带原行号的源码摘录](evidence/SOURCE_EXCERPTS.md)。下文的本机路径保留用于内部复现，不把那些文件当成已在 GitHub 发布。

## 结论

**千岛湖板端实际运行的是宽度 128 的 7,518,563 参数因果多束小波模型：`LateSpectralJointBeam(128) + WeakEchoVerifier`。** 骨干族标识 `causal_joint8_native_late_spectral_residual_research_v1`，冻结组合为 `signed_response@200 + range_balanced_identity + weak_verifier_original`。没有发现独立的对外语义版本号；复现应锁定三份权重、ONNX 与目标板引擎的摘要，而不是用“小波小模型”或单个文件名指代。

1. **PC 训练／离线推理：** PyTorch 组装三份 checkpoint。整航段 2,020 ping 的小模型外观图在工作站用同权重生成，不是板端导出的全幅图。
2. **转换：** `FP32LearnedStep` 将学习计算和四个状态导出到两个 ONNX。复数特征构造在引擎外；ONNX 不是原始 I/Q 到最终幅度的完整管线。
3. **Jetson 实跑：** 实采回放使用 R12189 引擎，仿真梳形使用 R12721 引擎，均为目标设备 TensorRT FP32 计划，经 CUDA Graph 逐物理 ping 调用。R12721 在 62,458 ping 主实验的 `model_SHA256`、板端准备校验和实际二进制一致；另有 103,055 ping `ready.digests.engine` 回执交叉确认。
4. **实时结论：** 支持有状态逐 ping 执行；主航程只记录模型段 P99 132.85 ms，不等于整链达 5 Hz。5 Hz 补充批次，包含恢复在内的板端去噪段 P99 约 240.00 ms，双臂配对墙钟吞吐 1.287 ping/s。没有单臂实装持续 5 Hz 的验收证据。
5. **论文勘误：** “32 ping 的板端推理回放”成立；逐样点 PC FP32↔NX TensorRT 增益误差最多 $3.43×10^{-5}$ 只核验了其中 **前 3 ping**。当前中文稿中“32 ping 的数值核验”应改成这个准确范围。此次不覆盖论文。

**运行不等于训练验收。** T1 的两臂自动选型 `eligible` 均为空；T1–T3 记录为开发输出、`automatic_export=false`、未消耗最终留出。后来独立的审阅、导出和 HIL 确实使用了三件对应权重，但这些事实不能反证模型已通过原自动质量门或独立航次验收。结论限定于有回执的历史板端运行，未核验今天板内服务状态。

### 证据及状态

- **已完成**：归档显示任务产物生成或运行结束，不自动等于精度/实时性验收。
- **已验证**：本审查已用摘要、实际日志字段、数值重算或代码严格加载核实；仅在注明范围内有效。
- **仅有代码**：可以找到实现，但没有绑定该具体工况的运行制品或日志。
- **尚未验证**：缺少原始证据或未开展对应测试，不以其他工况填值。

以下前缀均以仓库根目录为起点；`A`、`S`、`N`、`H`、`O` 是便于复核的本机索引，不是 GitHub 路径。指向本机但未提交的文件不在本次上传范围。

| 代号 | 位置 |
|---|---|
| A | `output/论文数据资料库_20261124/` |
| S | `A/01_声呐去噪模型/` |
| N | `A/02_匹配导航算法/` |
| H | `A/03_硬件在环平台/` |
| O | `output/online_sonar_student_20261118/qiandao_recorded_small_model_ab_prepare_v1/` |
| G56 | Git 已提交的 [56M 训练发布包](../../../research/sonar_denoising/random_adapted_56M_20261112) |
| G06 | Git 已提交的 [2026-11-06 冻结旧方案](../../../releases/sonar_denoise_20261106) |

| 证据 | 具体文件 / 关键字段 |
|---|---|
| W | `S/7_52M_模型权重/{wavelet.pt,spectral.pt,protection.pt,weights_sha.json}`；原始中转副本在 `O/weights/` |
| T1–T3 | `S/7_52M_实际选中模型_训练和评估账本/` 下 `小波骨干_signed_response_200.json`、`频谱读出_range_balanced_identity.json`、`弱回波保护_开发集对照.json`；分别按 `checkpoints[].SHA256`、`weights[].SHA256` 连接 W |
| E | `S/7_52M_ONNX与数值复核/{pc_recorded_preflight.json,full16_R12189_learned_fp32.onnx,full16_R12721_learned_fp32.onnx}`；同版 `O/pilot/` |
| J | `S/7_52M_NX实际TensorRT引擎/下载校验.json` 和两份 plan；板端原目录以 `qiandao_real_original_small_ab_20261119_v1` 结尾，R12721 在其 `comb_R12721_v1/` |
| P32 | `S/7_52M_NX实采推理核对/wired_ab_pilot.json`；32 条时序、3 条显式逐样点校验；实采原速及配对标志 |
| R | `N/实采1_40km_2020ping_同航次重放/{run_summary.json,trajectory.jsonl,actual_recorded_route_noisy_AB_audited_v2.json}`；`ready` 引擎指纹、2,020 ping/1,010 查询、25 次状态重置 |
| H0 | `N/完整梳形_62458ping_主对照/NX本次全程/{run_definition.json,run_summary.json,trajectory.jsonl}`；对应 `output/qiandao_historical_clock_ab_20261123_guard/full_v1/` |
| H5 | `N/完整梳形_5Hz_103055ping_补充对照/带条纹双臂/{run_definition.json,run_summary.json,trajectory.jsonl}`；运行副本 `O/nx_full_original_comb_noisy_ab_v1/` |
| B | `H/NX平台正式与失败记录/experiment_requirements.json` 的 `hardware_verified`，以及同目录 `NX_TRT_engine_identity.json`（后者是另一个 56M 引擎的记录，不是本次小模型构建日志） |
| F56 | `S/56M_NX全幅数值增益/run_summary.json` 内嵌 `original_actual_NX_report`；`S/56M_早期Jetson全幅输出/run_summary.json` 为 PyTorch 路径 |
| V56/V7 | `S/56M_工作站评估记录/evaluation/{run_summary.json,worker_*.json}` 与 `S/原生图_7_52M同权重工作站及对照/real_original_visual_review.json` |

### 制品指纹（本机重新计算 SHA-256）

| 制品 | 字节 | SHA-256 |
|---|---:|---|
| W / `wavelet.pt` | 30,077,404 | `76ac8d6792ef66cb402c86399cf0c2857469f563b3536df9b458be469d409723` |
| W / `spectral.pt` | 11,444 | `2c3ac2842ca8726f7d08b37c7859bef8d5c554b3766a3feb2b809b88c6449302` |
| W / `protection.pt` | 23,482 | `3fbf8d28f67f7e3f6d818507fede12715c43e98909d681cc6a6f533cf6c7a5c6` |
| E / `full16_R12189_learned_fp32.onnx` | 31,700,000 | `9d00f3bddb3268de014e02df865f050847876b1e8bdd3c0d1fcd412f4bc386c4` |
| E / `full16_R12721_learned_fp32.onnx` | 31,768,096 | `9a2b1a4bcab4a21a4d90e3dba1f9c0a43f71a45314294da81648c6ad727da117` |
| J / `full16_R12189_fp32_noTF32.plan` | 32,875,676 | `842d90945588a91c985e1db8402083aae0b1d933aaba324f3a49dce18585d587` |
| J / `comb_R12721_v1/full16_R12721_fp32_noTF32.plan` | 32,546,500 | `b20daf8809e304ab9a8f2032c391284b3ef1c837277dd4f2b8bd876135603bbf` |

H0 的 `experiment_source_SHA256` = `7bec4fdf69a946b6c0216187248157da97b12c007db830f46368beb527490a97`，等于本机 `standalone/RunQiandaoHistoricalClockAB.py`；其 `board()` 对匹配的 plan、板端源码和先验图做握手校验，回执写入逐 ping 数据。R、H5 的准备回执独立记录对应 plan SHA。**ONNX 与 plan 的完整编译命令及 TensorRT builder 逐层日志没有找到**，不能宣称已经从当前 Git 树一键重建出相同 plan。

### 已知候选排除

| 候选 | 定位与排除依据 |
|---|---|
| `fusion_wide_front`、`fusion_limited_front` | G06 两个已发布的旧 .pt；`standalone/RunFrozenSonarDenoising.py:143–187` 是该发布的离线加载与分块入口。与板端三个 W 摘要、`load_model()` 和 plan 摘要均不符。 |
| `MatchedSonarWavelet` | Git `standalone/MatchedSonarWavelet.py:19–42,72–91`；继承 BandAnchored，供前端带宽／方向性聚合对照，不是上述 ONNX 的类。 |
| `BandAnchoredNativeSonarWavelet` | Git `standalone/BandAnchoredNativeSonarWavelet.py:14–20`；继承 NativeLocal，独立离线候选。 |
| `NativeLocalSonarWavelet` | Git `standalone/NativeLocalSonarWavelet.py:17–66,80–100`；历史小波候选含方向残差融合及 2D 上下文。与实际板端按距离的 1D Haar／四状态接口不同。 |
| `PublicSonarEnvelopeModel` | Git `standalone/PublicSonarEnvelopeModel.py:13–41`；历史线性包络模型，输入签名不是同步 16 束与 4 组状态。 |
| `random_adapted_56M` | G56 公开代码，F56 证实也在 NX 离线全测线运行，用于离线恢复/先验场；不是在线模型。T1 记录离线教师哈希，但选中骨干训练目标中的教师权重为 **0**。 |

### 分卷

- [网络与推导](MODEL_ARCHITECTURE.md) · [端到端流程](PIPELINE.md) · [代码索引](CODE_INDEX.md)
- [性能与计时口径](PERFORMANCE.md) · [大小模型对照](MODEL_COMPARISON.md)
- [论文实验计划](PAPER_EXPERIMENT_PLAN.md) · [材料缺口](MISSING_MATERIALS.md) · [图源](figures/README.md)
