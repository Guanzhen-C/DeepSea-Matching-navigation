# 从原始采集到 NX 输出的研发、构建和运行流程

下表使用 [证据代号](README.md#证据及状态)。“实跑”指有归档记录；“仅有代码”或“尚未验证”不能替代日志。算子关系图见 [Mermaid 图](figures/README.md)。

## 1. 训练链：离线研究与冻结组合

| 环节 / 状态 | 代码位置；输入 → 输出 | 关键参数、依赖、记录及缺口 |
|---|---|---|
| 实采索引与数据接入 **已完成／已验证来源** | `RunFrozenSonarDenoising.py:55–140` 是历史 MAT 提取线索；小模型整幅运行 `RenderOriginalVoyageSmallModel172.run:69–90`、`QiandaoRecordedInputs.verified_inputs:22+` 从已冻结 `.npy` 及采集回执读取，不在 NX 重解 MATLAB。实际原生数组＋掩码＋时间戳 → 一 ping 16 路复回波。 | 实采 2,020 ping，原生单束 R=12,189。归档根 `S/56M_推理权重与16束实采输入/` 有 `manifest.json`、`source_metadata.json`；数据方限制，原始回波、GNSS、绝对位置不提交。补按航次/舷/束的去敏来源说明与解码者版本。 |
| 训练样本 **已完成** | `JointBeamSonarData.BeamGroups/grouped_example`、`JointBeamPairedFidelity.paired_example`、`CausalStudentCurriculum.plan_draw`。公开显示/包络/复数及本地采集索引 → 同舷 8 个物理槽的有扰动、无新增扰动配对张量与评分核心。 | 公共单视源不会复制成 8 路实测接收，缺槽使用有效掩码；实采保留全 8 个同舷束。记录按固定 draw 与 seed 对照。完整带哈希的去敏样本索引、划分和早期候选日志未入 Git；不把窗口数写成独立航次数。 |
| 条纹增强 **已完成／已核对代码** | `RandomRippleSonarData.example → RandomTriggeredSonarNoise.draw_scenario/apply_scenario`；`JointBeamPairedFidelity.paired_example` 组织已知 I/Q 新增扰动、幅度差和恢复监督。部署推理没有此步骤；仿真宿主在输入端另注入同一条纹，两臂共享。 | 多源弧形位置、相位、曲率、间距；支持区复数叠加，保护区不造扰动；显示/包络域不伪造相位。T1/H0 的 `noise_profile_SHA256` 为 `18e116f130944464e93a685f6120761042711fc9030d94139809fcea80ddf092`，等于 G56 [已提交的配置](../../../research/sonar_denoising/random_adapted_56M_20261112/configs/random_ripple_profile.json)。同 profile 不表示训练和 HIL 的随机场/seed 相同。 |
| 窗口特征、监督 **已验证代码** | `BaselineAugmentedJointBeam.student_tensors_with_baseline`、`CausalSonarFeatures.from_observed_log`、`JointBeamPairedFidelity.step_update`、`NativeSignedResponseLoss`。回波对 → 幅度、因果对比、真实相位差、掩码；已知新增扰动定义对数幅度修正目标。 | 同输入有噪/参考共享采样网格，不用未知真实纯净幅度作标签。T1 连续评分序列 32、burn-in 32、截断 8；这是其训练时间窗口，不是板端每次等待的 32 ping。原生距离裁窗由 draw/core 决定，不从生产引擎猜定训练分块。 |
| 骨干前代 **已完成／部分见证** | `TrainJointBeamCueColdPrefixFormal.py` 产生 T1 的既有因果骨干；`TrainNativeSignedResponse172.preflight/run` 冻结同初始化，作 control 与 signed_response 两臂。 | T1 `initial_checkpoint_SHA256/initial_state_SHA256`、`rank_receipts`。本轮四进程 NCCL，AdamW 初值 $3×10^{-5}$，cosine 至 $3×10^{-6}$，weight decay $10^{-5}$，step 0/50/200 验证；选 signed_response 第 200 步。离线 56M 教师仅在验证侧可用，本轮训练权重为 0。需补前代终点权重、精确环境和 rank 账本。 |
| 晚期频谱训练 **已完成／权重已验证** | `TrainLateSpectralRangeBalance172.design/fit/run`、`TrialNativeOrderedBeamPhase172.Readout`。固定骨干，缓存同 ping 频谱/增益/原生恢复对，训练晚期谱残差。 | 同噪声对比 nominal、range-balanced 和附加高亮尾部三支，128 步/臂，AdamW $3×10^{-4}$/weight decay $10^{-5}$。选 `range_balanced_identity`。T2 `checkpoints[]`、`cohort_ledger`、`scores`。验证仍为开发集。 |
| 弱回波保护 **已完成／权重已验证** | `NativeAmplitudeInnovation.AmplitudeHead` + `NativeWeakEchoVerifier`，上游 `TrialNativeWeakEchoVerifier172.main/fit`；`TrialNativeSplitFlankProjection172.main/fit` 对照记录 T3。 | 选择 T3 的 **`weak_verifier_original` 冻结原保护臂**，不是新增的 `merged_flanks` 或 `independent_flanks`。T3 52 fit、24 guard，两个分支以 384 更新试验；不要把 384 更新说成确选保护头的训练步数。T3 出具最终保护 .pt 指纹；保护前代链完整日志/中间权重尚缺发布。 |
| 独立验证 **尚未验证** | T1/T2/T3 含固定开发、保护或选择记录；`final_holdout_consumed=false` 是未用最终留出，不是已完成盲测。 | 共同比较测试集未找到完整四臂封存成绩。需补按航次、设备、噪声类型的盲测试，见 [实验计划](PAPER_EXPERIMENT_PLAN.md)。 |
| 权重冻结 **已验证** | `RenderOriginalVoyageSmallModel172.run:86–111` 按账本选择三件权重；`PrepareQiandaoRecordedSmallModelAB.load_model:28–39` 检三 .pt 哈希、CPU 安全反序列化并严格装载。 | W/T1/T2/T3 三摘要一一对应；原训练 `eligible` 空、未自动导出，冻结是后续明确的工程选择。组装不同于将 56M 量化或转成一个单文件。源码和制品未进基线 Git。 |

## 2. 模型转换和部署

| 环节 / 状态 | 代码、输入 → 输出 | 核对与缺口 |
|---|---|---|
| PC 同权重整段推理 **已完成** | `RenderOriginalVoyageSmallModel172.run`，`BenchFullNativeAmplitudeGraph172.stage`；PyTorch CUDA Graph、给定真实断点，并逐样本幅度增益＋既有色表出图。 | V7 的 2,020 ping、显示宽高 24378×8084、`render_seconds` 及三权重指纹；不是板端全幅显示、不是传感器实时验收。 |
| 两个静态 ONNX **已验证** | `PrepareQiandaoRecordedSmallModelAB.FP32LearnedStep.forward/export/pilot:42–153`。机器实值 7 入 5 出，四个显式状态；ONNX opset 17、静态形状、FP32、常量折叠。 | 本审查分别用 ONNX checker 校验两个二进制和绑定形状；E 中 3 个实采 ping 对比 PC PyTorch、ORT CPU；增益差最大 $1.90735×10^{-5}$，状态差最大 $1.41412×10^{-5}$。完整的 PyTorch/ONNX/ORT/CUDA 构建锁版本与终端日志仍缺。 |
| NX TensorRT 构建 **产物已验证，命令尚未验证** | J 两份实际 NX plan 已从板端存档；不是 NVIDIA AGX 引擎，也不是 `SonarNXTensorRT.build` 的 56M 分块接口。小模型 build 命令/脚本和逐层 tactic 未与两个生产 plan 哈希同绑。 | FP32 的计划内外绑定与日志都支持 FP32 路径；文件名及已有部署标签明示 `noTF32`，但缺该两 plan 的逐层精度和 builder flag 原始记录。不能从文件名独立证明所有 tactic；不能借 56M 或早期探针的 `trtexec` 命令充当证据。 |
| 目标板安装与身份校验 **已验证** | `LaunchQiandaoHistoricalClockAB.sh:13–22` 调用确切 R12721 plan；`RunQiandaoHistoricalClockAB.board:75–127` 比对引擎与依赖哈希；`RunQiandaoRecordedSmallBoardPilot.serve:214–235` 校验 R12189。 | J/H0/H5/R 的摘要及准备回执。部署启动环境（包清单／功率状态／工作线程配置）应与 plan hash 放在同一 manifest，而不混用历史海湖场景记录。 |
| TensorRT 逐 ping **已验证** | `RunQiandaoRecordedSmallBoardPilot.Runtime.__init__/_step/run:56–113`：deserialize、创建 context、校验 I/O 形状与 dtype、预热与 CUDA Graph 捕获，`execute_async_v3` 后将 4 个 next_state 写回，不走 PyTorch 模型回退。 | 计时从 `NativeLearnedGain.run` 的 CUDA Event 记录；图回放调用的张量拷贝／状态写回不是整个板端预处理与通信。实采在数据断点从零恢复因果状态、不会借未来 ping；R 记录 25 次复位。 |
| 原生幅度/复数输出 **已验证** | `RunQiandaoRecordedSmallBoardPilot.serve:238–280`、`RunQiandaoHistoricalClockAB.board:145–173`：用观察出的水柱/有效掩码乘 log 增益，再乘回 $A$ 或 $z$，分别把数组交给冻结导航或先脉压进入导航。 | 同输入验收用 P32/H0/H5/R；H0 62,458 条发生实际非零修改。不将 512 栅格、PNG 拉伸或色阶后处理算作恢复，不重新对实采接收 I/Q 施加原设备滤波。 |

### 环境的证据级别

- PC/训练：Python、NumPy、PyTorch CUDA/DDP/NCCL；T1 代码显式设置 TF32 off、cuDNN deterministic；两个后接分支另按各自开发账本。工作站确切驱动及每次实验的 pip/容器锁尚未完整归档。审查机器的 PyTorch 版本不反填训练服务器。
- PC 导出：PyTorch、ONNX、ONNX Runtime（E 中为 CPUExecutionProvider）、NumPy；PC 参考另在 CUDA 执行。
- 板端：TensorRT Python、PyTorch CUDA、NumPy、OpenCV，匹配部分另用冻结 LoFTR/Kornia 和 GTSAM / 项目导航模块。硬件环境、线程数与不同计时边界见 [性能](PERFORMANCE.md)。
