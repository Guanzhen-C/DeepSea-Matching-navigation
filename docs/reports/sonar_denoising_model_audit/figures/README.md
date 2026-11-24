# Mermaid 技术图

左舷和右舷各 8 束，总共 16 路。实线代表代码调用/数组流；虚线代表制品构建或同版验证，不表示原始数据上传。来源和冻结哈希见 [证据索引](../README.md)。

## 1. 板端推理结构（实际）

[可导入的 Mermaid 源](architecture.mmd)

```mermaid
flowchart LR
    A["原生一 ping 复回波<br/>2 舷 × 8 束 × R"] --> F["BaselineCueFeatureStream<br/>4 因果特征 + 独立有效掩码"]
    A --> S["NativeLocalSpectralCue<br/>同 ping Hann FFT → 24 谱特征"]
    A --> C["NativeCrossfitBeamCue<br/>同 ping 复数投影 → 6 cues"]
    F --> E["BaselineAugmentedJointBeam<br/>按舷分组 / 3 次距离 Haar / 4 个尺度"]
    H["前一 ping 的 4 组状态"] --> E
    E --> HN["4 组 next_state"]
    HN -->|"Runtime._step 回写"| H
    E --> B["基础 log 增益"]
    B --> L["LateSpectralJointBeam<br/>晚期谱残差，不写状态"]
    F --> L
    S --> L
    L --> W["AmplitudeHead + WeakEchoVerifier"]
    C --> W
    W --> G["16 × R 原生 log 增益"]
    A --> X["水柱保护下乘回原回波"]
    G --> X
    X --> N["实采恢复幅度 / 仿真恢复 I/Q 后脉压<br/>再进入既有导航"]
    subgraph TRT["FP32LearnedStep / TensorRT / CUDA Graph"]
        E
        B
        L
        W
        HN
    end
```

## 2. 训练、转换与两类板端实验

[可导入的 Mermaid 源](pipeline.mmd)

```mermaid
flowchart TD
    A["受限原生 I/Q + 公开授权训练源"] --> D["JointBeamSonarData.grouped_example<br/>保留真实同步束及支持"]
    N["RandomTriggeredSonarNoise<br/>draw_scenario / apply_scenario"] --> D
    D --> S["TrainNativeSignedResponse172<br/>骨干：signed_response@200"]
    S --> B["TrainLateSpectralRangeBalance172<br/>晚期头：range_balanced_identity"]
    S --> C["保护头迭代与冻结对照<br/>选 weak_verifier_original"]
    B --> C
    S --> W["load_model<br/>三件权重逐一校验"]
    B --> W
    C --> W
    W --> V["PyTorch 同权重整段可视化<br/>非板端"]
    W --> O["FP32LearnedStep.export<br/>opset 17 / R12189、R12721"]
    O -.-> P["PC preflight：ORT 比对 3 ping"]
    O -.->|"目标板实际计划已取证；完整构建命令缺档"| J["NX 两份 FP32 目标计划<br/>按 SHA 校验"]
    J --> T["RunQiandaoRecordedSmallBoardPilot.Runtime<br/>NativeLearnedGain"]
    T --> R["R12189 实采：32 ping 小试与 2020 ping 重放"]
    T --> H["R12721 仿真：62,458 / 103,055 ping"]
    R --> N1["幅度恢复 → 既有实采侧扫匹配"]
    H --> N2["复回波恢复 → 脉压 → 既有导航"]
```
