# 实际去噪输出接口：从复 I/Q 到幅度或复回波

证据 R/H0/H5/P32、S、F 的定义见[总览](README.md)。以下区分原生回波、学习增益、保护掩码与导航数据包。

## 1. 输入、输出与索引

| 项 | 实采 R／P32 | 仿真 H0／H5 |
|---|---|---|
| 来源 | 声呐设备已经接收并形成的波束复数记录；历史读入 `BeamSum` 的 8 束/舷（`RunFrozenSonarDenoising.extract:100–135`）。板端不从 PNG 构造 I/Q，也不补造相位。设备内部处理全部细节未知，不把该数据叫作阵元 ADC 原始采样。 | `SideScanComplexEcho.PointScattererIQSimulator.ping:171–214` 明示为等效波束形成、脉冲压缩前的复基带；合成散射、噪声和仪器式干扰，不是原设备实录。 |
| 线路格式 | 每 ping little-endian `<c8`，16×12,189 个 `complex64`；布局侧优先、束次之、距离连续（左 Port=0、右 Stbd=1，每侧束 0–7）。按原记录 `seq/record_index` 发送，另有两舷的 `<f4` gate 和元数据。板端解码 `(2,8,1,12189)`，模型统一按 `(2,8,R)`。 | 每 ping `<c8`、16×12,721 个 `complex64`，侧、束、距离顺序相同；仿真侧 0 对应本体横向非负一侧。壳层单独送序号/时间及运动事件；gate 板端从共同带噪 I/Q 计算。 |
| 引擎签名 | 统一 `features=(16,28,1,P)`、`positive=(16,1,1,P)`、`crossfit=(16,1,R,6)` 和四个状态。实值 FP32；R12189 的右端填充至 12,192。 | 同签名，R12721 的右端填充至 12,728。填充只为 Haar 对齐，距离轴不插值，输出裁回原宽。 |
| 模型输出 | `gain`：TensorRT FP32 `(16,R)`，设备绑定复制回 CPU 后 `np.float32 (2,8,R)`；同一 ping、舷侧、束、距离位置的一一对应有符号对数幅度增益。另回四状态，**不回传时间戳或测距值**。 | 同左；不是新的声学波形、测距数组或降噪图。 |

定位：`QiandaoRecordedInputs.verified_inputs:22–47`（原始签名、形状与真实时钟）；`RunQiandaoRecordedSmallBoardPilot.py:16–24,56–91,127–152,238–255`；`PrepareQiandaoRecordedSmallModelAB.py:42–91`（裁回原宽）；`RunQiandaoRecordedTrackAB.host:144–158`、`RunQiandaoHistoricalClockAB.board:147–167`（序号/时间戳/回执）。真实记录以整数微秒时钟相对首次采样重新定零，不是重新插成等间隔；每包核对摘要和严格时序。

## 2. 恢复公式与脉冲压缩的先后

设 $z_{s,b,k}$ 为接收复数，$A=|z|$，$g$ 为模型 log gain，$w\in[0,1]$ 为外部水柱保护权重。恢复只乘正实系数 $q=\exp(wg)$。

| 路径 | 真实操作 | 是否再做脉冲压缩 |
|---|---|---|
| R | 原始臂 $A$，处理臂 $A'=Aq$。代码先将量化后的入线 `complex64` 转 `complex128` 后取幅度，使用 `float64` gain 参与乘法；作为数值包输入导航而非导出复波形。 | **不做。** 原来怎样接收便从该设备记录定标继续处理；没有将幅度当复数重脉压。 |
| H0/H5 | 原始臂 $z$，处理臂 $z'=zq$。在乘法边界非零样点相位不变、掩码零处样点完全相同；其后各自按同一接收模型做 `pulse_compress(v)`。 | **两臂都做一次**与发射脉冲共轭时间反转的匹配滤波，`fftconvolve(..., mode="same")`，以 `complex64` 输出。它是接收处理，不是模型内部或第二次去噪。 |

源码：`RunQiandaoRecordedSmallBoardPilot.serve:255–272`；`RunQiandaoHistoricalClockAB.board:155–167`；`RunQiandaoNoisyCombBoardAB.board:96–112`；`SideScanComplexEcho.pulse_compress:216–221`。数学上乘增益与脉冲压缩**不交换**，不能用对压缩后幅度简单乘 gain 来替换现有 HIL。采样坐标不动并不保证脉压后的峰位和相位分布不变；卷积也可将支持区改动传播到邻近格，保护位相同的断言在**压缩前**，不是下游所有格。

## 3. 三种掩码不能混为一谈

1. **模型有效性 `positive`** 是原生有限 I/Q 中 $|z|>0$（`CausalSonarFeatures._step:48–83`），不是水柱真值。先按此支持制造对数幅度、相位差、局部对比等特征，引擎/入口保证零回波不产生有效增益；整个 gate 不是给引擎的输入张量。
2. **实采恢复 gate**：两条 `gate_{0,1}.npy` 由冻结原生包加载（`QiandaoRecordedInputs.verified_inputs:22–35`），每舷每 ping 一条，与同舷八束广播。来源 `real_observed_noise_support.npy` 的生成路径是 `RandomRippleEvaluation.prepare_real_support:36–52`：设备文件的底跟踪与首返回范围（`RunFrozenSonarDenoising.extract:123–129`）生成第一返回之后的 smoothstep；宽度 $\max(r_{\rm first}-r_{\rm bottom},\,c/2f_s)$；从门限零区出发，逐束把幅值不大于 $8/255$ 的连接暗区扩为共同保护区（`MeasuredStripeNoiseModel.noise_support_from_bounds:17–35`）。设备检测结果不是解析的水柱真值，也不是导航深度量测。实采用相同的 $(w>0.5)\land(z\ne0)$ 作两臂 **共同候选支持**；接收包另做幅度门限，因此最终有效格仍可因去噪而不同。
3. **仿真恢复 gate**：`OnlineSonarDenoising.observed_water_support:184–210`，按舷**按束**对 I/Q 功率取 65 格滑动均值；阈值 $\max(6\sigma_{\rm thermal}^2,\max|z|^2\times10^{-12})$；寻找连续 16 格过阈的首次位置，以后 64 格线性过渡到 1；无起点整行置零。板端由**同一个已加噪**输入计算，不从测线真值或底图推导，不直接滤波信号。宿主在加人工条纹前还用同类函数确定可注入支持；这与板端再从带噪输入估出的 gate 不是同一数组，不假定完全相同。

回波恢复后还要经过**导航测量构造**：512 格功率汇聚、检测掩码、特征图归一化和子图插值是下游，不发生在 7.52M 原生输出处。声学采样刻度、束方向和原记录时间戳没有被模型重写，但有效支持、强度峰、检测/匹配选择和分配的 source token 可以变，见[导航接口](NAVIGATION_INTERFACE.md)和[风险](POTENTIAL_ISSUES.md)。图像显示仍执行项目原尺寸约定，不把内部特征栅格当作原始声呐分辨率。
