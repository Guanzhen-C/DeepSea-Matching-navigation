# 配对数据、版本与验证边界

## 1. 复算方法与源文件身份

本审查逐行读 R、H0、H5 的 `trajectory.jsonl`，检查导航臂的 DR 序列逐输出相等；R 只统计真实 `query` 行并核对奇序号与原时间。按两臂各自非空的 `map_epoch` 在同一 ping 配对。汇总和轨迹文件的下表 SHA-256 为本次重算；本页未复制轨迹坐标、接收样点或 GNSS 数据。

| 对照 | `run_summary.json` SHA-256 | `trajectory.jsonl` SHA-256 |
|---|---|---|
| H0 | `c36920a8232492f2e9ca89c855dd44408cb9131a2633eba17181631cce7aada0` | `06e7dbb2f622c40b36161a168a66453d5d9ed4ec5985bfbc3ddbc3a6286f2943` |
| H5 | `ed9321fae386536694076d4ea6e0f801d5c0aba952e6fad8a73e94f39f875531` | `90214f2f7a869398be40fb6ef86713c9ae681c63c1839c2657d7bff51e87357a` |
| R | `c3ed5603d6dddd673b7e836d0ac80ac6af73256797de0f312101a079cd54dbf9` | `5efbff4bbe3568788a701bf54bcf6eaf7989da6e356a3e8059f52121c77cbc52` |

- H0 的 `run_definition.board_expected.code` 10 模块、H5 的 `run_summary.ready.digests.code` 9 模块、R 的 `run_summary.ready.code_SHA256` 9 模块均与 F 相应文件匹配。H0/H5 当前本机入口分别与各自 `run_definition.experiment_source_SHA256` 相符；R 不能据已校验的模块推断没有纳入回执哈希的每个外围脚本也做了同级绑定。
- F 的上一级 `deployment_manifest.json` 中 61 个 `runtime/` 文件均在档、逐文件 SHA-256 全部吻合。本地冻结包的文件完整性和板端握手绑定的子集是不同级别的证据，不把 61 个文件都写成每次握手单独核验。
- 冻结底图 H0 `backscatter.npy` 摘要以 `033cf9a3` 开头，H5 以 `3d507167` 开头（相应 `board_expected.prior` 与 `ready.digests.prior`）；**两个仿真批次的底图不同**，时钟、运动事件产生方式、接收背景干扰也不同（`RunQiandaoHistoricalClockAB.py:117–126,228–251`、`RunQiandaoNoisyCombBoardAB.py:26–28,159–171,202–213`）。不能跨批计算导航“改进”。

## 2. 公平性审查

| 项目 | R 同航次 | H0 原时钟 | H5 5 Hz 补充 |
|---|---|---|---|
| 起始输入 | 一份带真实噪声的纳入数据，原 ping 序号与时间；不额外加人工条纹 | 一份相同带噪复 I/Q（历史背景 + 额外条纹）；上位机独立校验历史原幅度 | 一份相同带噪复 I/Q、同噪声 seed；不是 H0 原时钟 |
| 地图 | 两臂同一份偶 ping 构造的 0.3 m 地图，用 56M 预处理；与奇编号无直接重号，但同航次不独立 | 两臂同一份冻结强度图 | 两臂同该批冻结地图 |
| 初值/运动 | `RecordedTrack.initial_state` 取同一个 header；两臂共享 GNSS 派生的航向/速度、名义水平与深度，不是真实 IMU/DVL/压力同步录制 | 共用历史初始状态和 1,339,709 个归档事件 | 同一仿真启动值、同一生成运动流 |
| 时间/比较参考 | 2,020 ping→同 1,010 奇编号查询；同航次 GNSS 水平参考 | 62,458 ping，约 0.330 s；共同上位机三维真值 | 103,055 ping，0.2 s 仿真间隔；共同上位机三维真值 |
| 两臂 DR | 1,010/1,010 个查询位置逐项相等 | 62,458/62,458 个位置相等；宿主还逐 ping 核验与归档 DR 的 bitwise 一致 | 103,055/103,055 个位置逐项相等 |
| 算法版本 | 两臂都用 `adaptive_projected/mapped_range_changes/preserve_legacy`，不切换到旧导航 | 同左；实际运行加了两臂共用的候选无效斜距防护 | 同左导航开关，**不是** H0 带 guard 的运行入口 |

结论：**原生输入完全相同**可以实现，且已有三次记录。准确表述是“分流前同一 I/Q、同一运动流、同一先验、同一初值与时钟，唯一处理因素为去噪开关”；**导航包不可能在干预有效时仍完全相同**。实采两臂初始候选掩码相同但最终幅度阈值可能不同，仿真脉压与检测也会改变有效格。这属于干预经适配器自然传播的效果；如需单独量化幅度的影响，应另做同支持消融，不在主试验强制抹掉这些变化。

## 3. 防误判的三个细节

1. **历史基线不是另一个导航版本。** `historical_no_extra_ripple` 或 H5 `no_extra_ripple` 自带不同的噪声条件；不能拿来填四臂 A 或 B。四组实验的固定因素见[方案](FOUR_ARM_EXPERIMENT.md)。
2. **R 同航次有隐含相关性。** 偶/奇只保证原 ping ID 不复用；共同地物、相邻时间、同一 GNSS 受力与轨迹代理仍相关。因果模型在奇 ping 计算时还可能利用之前偶 ping 的历史状态，这与 A/B 共享数据源的内部对照并不矛盾，但进一步限定了“测试独立”的说法。新试验建议用独立测线或航次建图。
3. **H0 候选修复不是性能消融。** 归档 `interrupted_prefix_equal_input_pings=19499`、前缀位姿最大差为 0，完整运行有 1 次 raw、0 次 student 因非法观测拒绝（`rejected_loop_candidates_by_arm`）。施工防护对两臂一致，大约 5.148 km 的触发不是挑真值误差删除。不能把前段与修复后的后段混作旧/新导航对照。新四臂共同冻结相同无效输入规则，不把异常处理混作所提算法的效益。
