# 单物理 Ping 固定滞后声学前端验证报告（2026-09-15）

## 结论

正式前端已从“连续 8 个物理 ping 的滑动观测”改为“每个 5 Hz 物理
ping 只作为一次当前观测”。一次 V2 ping 内原有的每侧 8 行仍是同时采集
的空间行，并未删减。当前 ping 只匹配因果的 30 s 历史 surfel 子地图，
完成更新后才进入地图；过期批次随后从体素统计量中精确移除。

受控长程实验表明，单 ping MBES 直接写 yaw 会产生系统性航向偏差。因此
正式 Local 使用 position-only Schmidt 更新；IMU/DVL/深度计维持局部姿态、
速度和偏置，Global 使用跨 ping 的测深地形与侧扫纹理估计绝对 x/y/yaw。
`position-yaw` 和 `full` 策略仍保留为实验选项，但不是正式默认值。

## 数据归属与相关性处理

- 物理采集率为 5 Hz，航速 1.5 m/s，相邻 ping 前进约 0.30 m。
- 一个 ping 的沿航向覆盖约 1.20 m，因此相邻覆盖约 75%。所有几何残差
  均保留，但信息矩阵连续乘以新覆盖比例，实测全段均值约 0.247。
- 30 s 固定滞后窗口在稳态保存 151 个边界采样（浮点事件时刻包含两端），
  对应约 45 m 轨迹；当前 ping 不在自己的参考地图中。
- 测深 XYZ 与侧扫强度来自同一原始返回网格。正式 Local 只由 MBES 几何
  写一次状态；侧扫直接匹配以 graph-only 方式保留，不第二次写 Local。
- 未新增基于真值、位置误差或残差大小的选择门限。

## 数值实现

固定滞后删除采用可加减的体素一阶/二阶充分统计量。每个体素以自身原点
为数值参考，统计偏移量而不是直接对数十万至数百万米的 UTM 坐标做大数
相减。320 ping 的 UTM 滑窗单元回归中，连续保留 150 ping 后的增量地图与
把剩余数据从头重建的地图一致。

## 30 km 起点、种子 31 的 A/B 结果

所有 Isaac Sim 实验使用同一公开 Monterey Canyon 场景、同一 5 Hz 集成
声呐、同一 IMU/DVL/深度事件时钟、同一起点真值初始化以及相同 Global
GTSAM/cohort-particle/地形匹配配置。误差在同步的 5 Hz 声学时刻计算，未做
轨迹对齐。

| 配置 | 距离 | DR RMSE | Local RMSE | Local 相对 DR | Local 末端误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 7a995：滑动 8 ping，full | 2 km | 14.233 m | 5.136 m | 改善 63.91% | 14.216 m |
| 非重叠 8 ping，position | 2 km | 14.233 m | 11.884 m | 改善 16.50% | 21.935 m |
| 单 ping，position+yaw | 2 km | 14.233 m | 27.073 m | 恶化 90.22% | 56.245 m |
| 单 ping，full | 0.5 km | 2.242 m | 2.205 m | 改善 1.64% | 4.133 m |
| 单 ping，position（正式） | 0.5 km | 2.242 m | 1.921 m | 改善 14.31% | 2.977 m |
| 单 ping，position（正式） | 2 km | 14.233 m | 8.185 m | 改善 42.49% | 13.623 m |

旧 7a995 的 Local RMSE 数值仍更低，但同一个物理回波最多进入 8 个连续
非线性状态更新，`1/8` 信息缩放只能修正协方差强度，不能撤销重复的状态
均值写入，因此该数字不作为单次观测一致性的证明。新的正式版牺牲了这部分
表面收益，换取每个物理 ping 唯一归属和有界的因果参考地图；其最终 Global
RMSE 为 1.160 m，也低于该旧对照的 1.303 m。

2 km 正式结果的 Local/DR 100 m SE(3) 平移 RPE 为 1.472/1.526 m，
1 km 平移 RPE 为 16.802/20.041 m。Local/DR yaw RMSE 为
0.475/0.545 deg。Global 在线位置 RMSE 为 2.016 m，最终固定滞后图结果为
1.160 m；其用途是绝对地形锚定，不能与 Local 相对里程计混为一项指标。

正式 2 km 运行处理 6667/6667 个物理 ping，接受 5420 次几何更新；窗口
最终保存 151 ping，累计淘汰 6516 ping 和 1,262,399 个插入点，地图保留
28,822 个点。侧扫有 2960 次合格 graph-only 匹配、0 次 Local 状态写回。

## 复现结果

- `scientific_single_ping_smoke_start30km_120m_20260915.json`
- `scientific_single_ping_regression_start30km_2km_20260915.json`
- `scientific_single_ping_fullstate_ab_start30km_500m_20260915.json`
- `scientific_single_ping_position_ab_start30km_500m_20260915.json`
- `scientific_single_ping_position_regression_start30km_2km_20260915.json`

完整软件测试为 `356 passed`，并通过 Python 编译、启动脚本 Bash 语法和
`git diff --check`。120 m 端到端冒烟和 2 km 正式回归均以
`requested_survey_segment_reached` 正常结束。

## 科学边界

这些结果是确定性仿真，不是海试精度。地形来自公开实测栅格，但回波是
射线与强度模型合成数据；尚未覆盖声速剖面折射、多径、气泡、时钟偏差、
DVL 丢底及外参时变。2 km 结果证明当前种子和航段未发散，不等于完整
72.35 km 航线或多海域统计结论。完整路线需要另行连续运行并审计分段
RMSE、RPE、重定位成功率及失效区间。
