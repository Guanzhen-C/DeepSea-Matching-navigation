# 板端 7.52M 模型结构与数学表达

证据代号见 [README](README.md)。以下结构按本机冻结加载入口和源文件核对，不把类似文件名视为相同模型。

## 1. 三个输入层级

| 层级 | 接收 / 几何 | 数值含义 |
|---|---|---|
| 设备接收 | `(2,8,R)`，一 ping 左右各 8 束，`complex64` | 实测或仿真已接收 I/Q；实采根档支持更高存储精度，运行端明确转换为 complex64。不是 PNG，也不是只给强度。 |
| 外部特征 | `features (16,28,1,P)`，`positive (16,1,1,P)`；`crossfit (16,1,R,6)` | 4 个骨干特征＋24 个同 ping 谱特征；独立有效掩码和复回波辅助投影特征，各用 FP32。4 组历史状态另外输入。 |
| 学习输出 | `gain (16,R)`，另回传 4 组 `next_state` | 逐束逐距离格有符号对数幅度增益 $\Delta$，不是噪声波形，也不是恢复后图像。 |

$R=12,189$（实采）或 $12,721$（虚拟接收）；$P=8\lceil R/8\rceil=12,192$ 或 $12,728$，仅右侧复制填充 3 或 7 格；结果只写回原生 R。引擎对完整 16×R **不分块、无 Halo、无块间滑动步长**；时间调用步长是一物理 ping。下面的 65／32／24 等是特征上下文，不是离线推理 Halo。既有 512 格仅是匹配导航下游。

## 2. 特征构造

记 $z_{b,t,r}$ 为一束的复回波，$M_+=1[|z|>0]$。在有效点 $q=\ln |z|$、$u=z/|z|$，其余为零。当前行加过去最多 4 行、距离轴 65 点（±32）对有效 $q$ 取归一化局部均值 $\mu_+$。骨干四特征为 $M_+(q-\mu_+)$、相邻有效距离格 $u_r\overline{u_{r-1}}$ 的实部与虚部、以及由当前 $q$ 减去第一个对比通道重建的基线量；代码保留先减后减的两次 FP32 舍入，不能偷换成任意外部均值文件。缺乏实测相位的公开训练域用零相位特征；板端实采走 domain=3。见 `CausalSonarFeatures.py:7–24,48–98`、`BaselineCueFeatureStream.py:9–25`、`CausalBaselineCue.py:11–15`。

同 ping 频谱分支：移动 32 点 Hann 窗，距离步长 1，复数 FFT 划为 8 个频带；每带输出自身归一化功率、排除自身的同舷同位置功率均值与标准差，共 24 通道。窗口有效计数和同舷有效束不足时对应特征置零。频谱只作为特征，**不会删去频带或先滤波 I/Q**。`NativeLocalSpectralCue.py:7–47`。

保护头的 6 维 `crossfit` 是同 ping 复数辅助量，由同舷对束和距离近邻外侧区的投影、局部功率、相位坐标以及置信/计数等产生。基础投影 `RADIUS=24`、`GUARD=4`、最小上下文 14；它们不等于 T3 候选实验的 40/6（后者用于新增试验分支）。部署的是原保护头，不用新增两个 split-flank 学习分支。见 `NativeCrossfitBeamCue.py:6–61`、`NativePairedComplexInnovation.py:10–31`、`PrepareQiandaoRecordedSmallModelAB.py:58–71`。

## 3. 编码、递归与解码

- 以同舷 8 束分组；两个舷侧独立拥有状态。域仿射调制每束前三特征。头部把 8×3 特征与 8 个有效支持（合计 32 通道）做距离 $1×7$ 卷积；8 个基线通道经无偏置 $1×1$ 分支后相加。
- 宽度 $128,256,384,512$；四个状态 $h_i$ 的形状 $(2,128(i+1),1,P/2^i)$，$i=0,1,2,3$。每个尺度有 2 个空间残差块（深度卷积核 7、膨胀 1 和 2，逐点通道混合）与 1 个门控递归块。**3 次一维距离 Haar，4 个编码尺度**，不对时间做下采样。相邻距离格 $a,b$ 变为 $(a+b)/\sqrt 2,(a-b)/\sqrt 2$，经 $1×1$ 混合投影至下一尺度。
- `CausalFusion` 计算 reset/retain 门，$\tilde h=\tanh C_3([x,\mathrm{reset}\odot h_{t-1}])$，$h_t=M_{\rm side}[(1-\mathrm{retain})\odot\tilde h+\mathrm{retain}\odot h_{t-1}]$。解码分 3 个逆 Haar 级，逐尺度加跳接并做 1 个空间残差块；尾部 $1×7$ 输出每舷 8 条不同增益，掩码仍按束保留。可复核的“层数”应写为四尺度、8 个编码和 3 个解码空间残差块、4 个递归单元，而不是把图算子条数称为网络层数。
- 没有 `NativeLocalSonarWavelet.SparseDirectionalNoiseConsensus` 或历史 `MatchedSonarWavelet` 的方向性残差融合；有深度可分离**距离**卷积、同舷融合、重复因果状态与同 ping 频谱／复数 cues。方向谱 cue 与旧版多方向残差融合不是同一概念。

见 `CausalSonarWavelet.py:17–73`、`JointBeamCausalSonarWavelet.py:20–70`、`BaselineAugmentedJointBeam.py:31–86`。本机 CPU 严格加载三份权重后，`sum(p.numel())` 为：骨干含基线 7,512,008，谱读出 2,049，保护分支含幅度父读出 4,506，合计 **7,518,563**。该数为 PyTorch 名义参数（包含部署固定域后可能常折叠的域参数），不是 ONNX initializers 或引擎显存。

## 4. 晚期读出与恢复

晚期读出 MLP 为 29→32→32→1（`SiLU`，最后 $0.5\tanh$）：输入是 4 个骨干特征的逐点 tanh、基础增益 $g_0$ 的 tanh 和 24 个谱统计。有效点 $g_{\rm late}=g_0+M_+a_\theta$；晚期分支不更新骨干状态。随后 `AmplitudeHead` 将 $g_{\rm late}$ 与特征和复数 cue 变为父增益 $g$，`WeakEchoVerifier` 再预测保留分数 $s$，输出

$$
\Delta=g\big[1-m(1-\sigma(s))\exp(-(g/0.22)^2)\big],\quad m=1\ \text{（冻结权重缓冲区）}.
$$

缩放系数介于 0 与 1：保护头相对于其父增益 $g$ 不翻符号、不放大，但这不推出真实纹理无损，也不约束前面的谱/幅度父分支进行何种修正。见 `LateSpectralJointBeam.py:13–36`、`TrialNativeOrderedBeamPhase172.py:58–68`、`NativeAmplitudeInnovation.py:8–32`、`NativeWeakEchoVerifier.py:10–32`。

引擎输出之外，实采调用 $A'=A\exp(M_{\rm water}\Delta)$ 并以幅度交给匹配；仿真恢复复回波 $z'=z\exp(M_{\rm water}\Delta)$ 后沿既有接收链做脉冲压缩和导航。禁止区乘子为 1，可观测点配套独立的保护掩码；正实增益保持输入相位，不表示恢复无噪声相位。不对输出图拉伸或插值增样。详见 [流程](PIPELINE.md) 和 [结构图](figures/architecture.mmd)。
