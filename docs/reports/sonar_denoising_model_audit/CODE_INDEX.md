# 代码索引与入口分流

行号以审查时工作区为准。标记 **G** 表示可在基线 Git 找到；**L** 表示当前本机工作区 / 归档可读，但不是基线 Git 可复现的文件。除另注明外，位置均以 `standalone/` 为前缀。[主入口](README.md) 记录相应制品及运行摘要的本机位置。

## 1. 实际工作流

| 步骤 | 代码 / 调用点 | 状态和输出 |
|---|---|---|
| 历史输入解码参考 | **G** [`RunFrozenSonarDenoising.py:55–140`](../../../standalone/RunFrozenSonarDenoising.py#L55) 的 `inspect_source/extract`；**L** `QiandaoRecordedInputs.py:22+` 的 `verified_inputs` | 区分历史 MAT 读取与实际小模型运行读取已冻结 I/Q/mask/时序，不能用 PNG 恢复 I/Q |
| 训练窗口 | **L** `JointBeamSonarData.py:24–125` 的 `BeamGroups/grouped_example`；`JointBeamPairedFidelity.py:28–48,67–100+` 的 `paired_example/loss_terms/step_update`；`CausalStudentCurriculum.plan_draw` | 配对、同一非新增参考、同舷槽位、严格掩码 |
| 已知额外困难干扰 | **L** `RandomRippleSonarData.py:14–48` 的 `example` → `RandomTriggeredSonarNoise.py:22,144` 的 `draw_scenario/apply_scenario`；相同核心函数家族 **G** [56M 发布树](../../../research/sonar_denoising/random_adapted_56M_20261112/standalone/RandomTriggeredSonarNoise.py) | 全局坐标弧形扰动；训练 exact profile 用 T1 摘要绑定，不以 Git 发布版推定逐字相同。仿真 `OnlineSonarDenoising.StreamingRippleInjector` 的逐 ping 插入由宿主调用 |
| 原生因果输入 | **L** `CausalSonarFeatures.py:7–24,48–98` 的 `from_observed_log/CausalFeatureStream`；`BaselineCueFeatureStream.py:9–25` | past4+current、65 距离格、相位差、基线 |
| 骨干与小波 | **L** `CausalSonarWavelet.py:17–73` 的 `range_haar/inverse_range_haar/SpatialBlock/CausalFusion`；`JointBeamCausalSonarWavelet.py:20–70`；`BaselineAugmentedJointBeam.py:31–86` 的 `step` | 两个同舷组、4 个状态、3 次距离下采样 |
| 同 ping 谱与保护 | **L** `NativeLocalSpectralCue.py:7–47` 的 `_local_spectra_compute`；`NativeCrossfitBeamCue.py:6–61` 的 `side_flanks/projection_compute`；`NativePairedComplexInnovation.py:10–31` 的 `coordinates/project_planes` | 24 谱特征、6 复数辅助特征；不改动 received 波形 |
| 晚期和恢复预测 | **L** `LateSpectralJointBeam.py:13–36` 的 `step`，`TrialNativeOrderedBeamPhase172.py:58–68` 的 `Readout`，`NativeAmplitudeInnovation.py:8–32` 的 `AmplitudeHead`，`NativeWeakEchoVerifier.py:10–32` 的 `from_parent/forward` | 晚期残差、父幅度读出和弱增益保护；原生 log 增益 |
| 实际选模训练 | **L** `TrainJointBeamCueColdPrefixFormal.py`（前代），`TrainNativeSignedResponse172.py:40–193`（骨干实际选择），`NativeSignedResponseLoss.py:16–63`（目标），`TrainLateSpectralRangeBalance172.py:22–178`（晚期头），`TrialNativeWeakEchoVerifier172.py:96–183` 及 `TrialNativeSplitFlankProjection172.py:26,129–166`（保护实验与冻结对照） | T1/T2/T3 有对应运行账本，前代完整依赖闭包尚待包化 |
| PC 同权重整段可视化 | **L** `RenderOriginalVoyageSmallModel172.py:69–208` 的 `run` → `BenchFullNativeAmplitudeGraph172.stage` | V7；独立于 NX 部署，不将 16 ping 主机 I/O 分批说成并行改写模型状态 |
| ONNX 导出及 PC 检查 | **L** `PrepareQiandaoRecordedSmallModelAB.py:28–153` 的 `load_model/FP32LearnedStep/cues/export/pilot` | W → 两个 E；导出学部分；编译命令仍缺 |
| 板端实采 | **L** `RunQiandaoRecordedSmallBoardPilot.py:56–152,214–276,283–373` 的 `Runtime/NativeLearnedGain/serve/run`；`RunQiandaoRecordedTrackAB.py:22–87,101+` 的 `RecordedTrack/host` | P32、R；前处理在 CUDA、TRT 图与状态、实采恢复幅度，次序与断点核验 |
| 板端完整梳形 | **L** `LaunchQiandaoHistoricalClockAB.sh:13–22` → `RunQiandaoHistoricalClockAB.py:75–177,182–352` 的 `board/host`；H5 的 `RunQiandaoNoisyCombBoardAB.py:31–119,121–249` | H0 工况复用历史时钟，H5 独立补充；两个入口共调用同一 `Runtime` 与 `NativeLearnedGain`；仿真 $z'$ 接收脉压及导航 |
| 导航入口（与识别模型分开） | **G** [`SideScanIQHistory.py`](../../../standalone/SideScanIQHistory.py) 的 `packet_from_compressed_IQ`；冻结运行代码见 `N/复现实验_冻结导航运行时/runtime/`，含 `PureSideScanFeatureNavigation.py` 等 | 512 格栅格位于这里；H0 `board_expected.code` 逐项锁子版本，不以主分支同名文件覆盖 |

## 2. 近名入口不能互相代用

| 只用于识别历史或原型的入口 | 不应拿来替代的结论 |
|---|---|
| **G** [`RunFrozenSonarDenoising.load_model`](../../../standalone/RunFrozenSonarDenoising.py#L143) 固定 G06 `fusion_wide_front`（step 400、learned_fusion） | 已冻结发布，但不绑定两个 J 哈希 |
| **G** [`MatchedSonarWavelet`](../../../standalone/MatchedSonarWavelet.py#L19)、[`BandAnchoredNativeSonarWavelet`](../../../standalone/BandAnchoredNativeSonarWavelet.py#L14)、[`NativeLocalSonarWavelet`](../../../standalone/NativeLocalSonarWavelet.py#L53)、[`PublicSonarEnvelopeModel`](../../../standalone/PublicSonarEnvelopeModel.py#L13) | 历史二维切块／包络及候选实现，不能从“轻量”推定是板端配置 |
| **G** [`NoFIRSonarMWCNN`](../../../research/sonar_denoising/random_adapted_56M_20261112/standalone/NoFIRSonarWavelet.py#L69) + **L** `JetsonSonarDeployment`、`SonarNXTensorRT.build/TensorRTBackend` | 56M 离线分块和历史计划；其动态 tile I/O、Halo 都不属于固定 R 的小模型 |
| **L** `CueStudentRuntime.CueDenoiser/load_trial`、`BenchmarkSyntheticIQCueTensorRT`、`VerifySyntheticCueTensorRT`、`JointBeamCausalDeployment` | 早期端侧设计与探针，除非权重和 plan SHA 与 W/J 一致，不作为此版本 TensorRT 配置或性能证据 |
| Git **G** [`demo/hil/jetson_20261021`](../../../demo/hil/jetson_20261021)、[三场景报告](../jetson_hil_research_bundle_20261112) | 证明历史导航 HIL，不证明该批去噪权重已运行 |

## 3. 再核验入口

- Git 可见性：`git ls-tree -r --name-only eb5021a`；只看工作区或 grep 发布说明会将未提交权重和已提交代码混淆。
- 文件完整性：对 W 的三 `.pt`、E 的两 `.onnx` 和 J 的两 `.plan` 分别运行 `sha256sum`；逐项对照 [制品表](README.md#制品指纹本机重新计算-sha-256)。对本机 H0 代码执行 SHA-256，须等于 H0 的 `experiment_source_SHA256`。
- ONNX：`onnx.checker.check_model` 及 `graph.input/output` 形状表；本次确认两者都为 17、全静态 Float32，未重新导出或覆盖任何模型。
- 本机 PyTorch CPU 严格装载：`PrepareQiandaoRecordedSmallModelAB.load_model(weights, "cpu")` 后分别累加网络和保护头的 `p.numel()`；实测 7,518,563，`head.mix=1.0`。没有在此机器拿外部未知模型运行。
