# 仓库资料缺口与补交清单

以 `eb5021a` 为仓库可见基线。**本机有**不等于 Git 已提交；**未找到**不等于从未生成。前缀通过 [证据索引](README.md#证据及状态) 展开。建议建立独立的小模型发布包，将代码、二进制 SHA-256、环境和运行回执统一绑定；当前任务不上传权重和原始数据。

## A. 必须补充：最低可复核部署包

| 项目 | 状态 / 实际位置 | 尚缺原因和建议格式 |
|---|---|---|
| 实际权重 | **本机有／Git 无**；W 三 `.pt` + `weights_sha.json`；G06 已入库的两权重不是现用模型 | 打包原始三权重并附许可、大小、逐件 SHA，按授权放私有 Release 或受控制品库。重新转 safetensors 时另记转换与张量 SHA，不替换历史文件指纹。 |
| 架构与工具闭包 | **本机有／Git 未完整提交**；`standalone/`、`S/7_52M_相关架构训练与部署源码/` | 归档仅一部分源文件；须包含 `NativeAmplitudeInnovation`、`NativePairedComplexInnovation`、`CausalSonarFeatures`、`BenchFullNativeAmplitudeGraph172` 及训练/导出依赖闭包。建议独立包、环境锁、CPU smoke。 |
| 推理和启动代码 | **本机有／Git 无**；`RunQiandaoRecordedSmallBoardPilot.py`、`RunQiandaoHistoricalClockAB.py`、`RunQiandaoNoisyCombBoardAB.py`、`RunQiandaoRecordedTrackAB.py`、启动脚本 | 不用旧导航 HIL 代替本模型。提交去除私网/账号与绝对路径的入口和 smoke；保存对应历史源码 SHA。 |
| 配置和数据契约 | **部分存在**；W/E/J、H0/H5/R 的 `run_definition/ready`、`PrepareQiandaoRecordedSmallModelAB.py:21–91`；本机 H0 调用 `configs/online_sonar_hil_nx16_causal_joint8_wired_v42.json`（联机启动配置，不是本模型网络配置） | 补固定 R/P、7 入 5 出、dtype、时钟/断点、掩码、线程的 YAML/JSON 与校验器；从本地联机配置导出不含私网或帐号的模板。 |
| ONNX / TensorRT | **本机有／Git 无**；E 两 ONNX、J 两实际目标板 plan | 私有制品库或授权附件保存原文件和 SHA；仓库提交构建与下载声明、运行设备兼容性。56M 超过普通 Git 单文件容量的制品走适当制品库。 |
| 板端运行日志 | **本机有／Git 未收录对应版**；P32/H0/H5/R 的 run summary / ready / trajectory，各有 stderr 归档（空文件不证明构建过程） | 提交脱敏 summary、回执和只含序号/相对时间/计时/计数的 CSV.gz。不要提交 I/Q、独立 GNSS、绝对坐标、私网、账号或全量私有日志。 |

## B. 重要补充：复现和性能

| 项目 | 状态 / 实际位置 | 建议格式及原因 |
|---|---|---|
| 训练数据说明 | **部分存在**；T1 的 source/acceptance SHA，G56 DATA_CONTRACT 不是小模型数据清单 | 按公开/受限、域/槽位、航次匿名组、许可证及 split 提供 CSV/JSON；受限原始回波只做内部引用。 |
| 训练配置 | **部分存在**；T1–T3 与本地训练脚本 | 记录父模型ID、三阶段 YAML、打分阈值、正式 noise profile SHA、draw/seed、依赖环境。明确最终采用的保护原臂不等于对照分支 384 步。 |
| 训练过程记录 | **本机有摘要；前代记录不齐**；T1–T3 | 补完整训练步数日志、配对 rank receipt、validation、前代终点和参数/优化器配置；公开开发成绩时注明掩码与共同评估集。 |
| 模型转换记录 | **E 导出已验证；确切生产 build 缺** | 同版 NX 构建 JSON、实际构建命令、TensorRT/版本、tactic/workspace/精度、FP32/TF32 开关、I/O/层精度与两份 ONNX、plan SHA；不能以旧 56M 或候选模型命令替代。 |
| 资源、加载与持续时序 | **该小模型同运行证据缺失**；B/历史 tegrastats、F56 是其他工况 | 追加同步采集 tegrastats、进程 RSS、TRT/CUDA allocator、stages_ms CSV.gz 与温度/功率；区分装载、图捕获、首帧和热稳态；给出截止期、积压、丢包。 |
| PC↔NX 一致性 | **窄范围已验证**；E、P32 的数值回传各仅 3 ping | 补两个 R 的多段连续/复位对照、原生 gain/state/恢复摘要、配置 hash 与可公开合成 fixture；不能上传本机验证用实采 I/Q。 |

## C. 论文实验补充

| 项目 | 当前证据 | 建议补交 |
|---|---|---|
| 去噪前后数据 | 本机 V7/V56 的图和部分数值资料，原始实采回波受限 | 根据授权给衍生谱/统计、合成 fixture、等比预览；源 .mat、I/Q、GNSS 和未经批准的原片不入库。 |
| 独立测试集 | 开发／保护集、同航次和仿真；T1–T3 未消耗最终留出 | 新航次/设备/底质，匿名 split 与采集质量，保留测试前冻结与留出策略。 |
| 纹理保持 | 56M 身份误差、小模型开发样例与保护代数性质；无同域同条件两模型分数 | 共同地标、阴影、弱回波与保护区的保持指标，报告过抑制和伪影，预注册非劣门槛。 |
| 特征匹配 | P32 小样本相对对照，无独立几何的四臂 | 做跨重访真值标注，记录每窗精度/召回、无结果与残差，以配对法统计。 |
| 导航精度、复杂度 | H0/H5/R 有旁路/小模型；无四臂同条件；FLOPs 不齐 | 补同地图、真值、运动、初始化的四臂和实时单臂验收；发布 FLOPs/MACs 口径及结果。 |

优先提交 A 的**三份权重、两份 ONNX、两份目标板 plan、实际运行/导出源码、环境与关键回执**。第二步补同计划的构建与资源日志，第三步完成独立四臂验证。公开范围由数据和模型权利人确认；文档与合成夹具可以先行。
