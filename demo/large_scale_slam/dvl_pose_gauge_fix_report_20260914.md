# DVL 可观测性修正与回归报告（2026-09-14）

## 结论

DVL 是此前前端偶发突跳的直接触发器，但不是因为模拟 DVL 数据出现了异常值。根因是标准 ESIKF 更新通过长期积累的交叉协方差，让一次正常的体坐标速度残差同时大幅回拉累计位置和全局航向不可观方向；声呐几何退化时，这个问题尤其明显。

本次修改把每次 DVL 更新限制在它实际提供信息的子空间内：保留三轴速度、IMU 偏置以及可观的横滚/俯仰修正，保持累计位置和全局航向规范方向不被单次 DVL 速度观测重置。没有新增残差门限，也没有关闭或降频 DVL。

同时修复了误差状态注入后的协方差重置顺序：现在先形成 Joseph 后验协方差，再用误差状态 reset Jacobian 将其变换到新的名义状态切空间。旧实现计算过 reset，随后却被未变换的 Joseph 协方差覆盖。

## 故障复现证据

- 起点：全地图路线 30 km 位置；随机种子：31；真值初始化。
- 原实现到路线约 33.507 km 时，一次仅 0.0986 m/s 的普通 DVL 残差造成 4.967 m 位置跳变和 1.011° 姿态跳变。
- 该次 DVL 马氏距离为 4.21，测量本身正常；异常来自位置增益范数已增长到约 52.8 s。
- 同一区域声呐平移信息的归一化特征值约为 `[0.00073, 0.00195, 1.0]`，弱方向接近 AUV 前向，且路线俯仰接近 -11.91°，符合“弱声呐约束放大滤波器不可观一致性问题”的现象。

## 确定性 IMU/DVL/深度 A/B（4.5 km）

两套滤波器接收完全相同的测量；本试验不加入声呐或地形匹配，用于隔离 DVL 修正方式。

| 指标 | 未约束 DVL 修正 | 可观测子空间 DVL 修正 | 变化 |
| --- | ---: | ---: | ---: |
| 位置 RMSE | 71.655 m | 39.376 m | -45.1% |
| 最大位置误差 | 139.215 m | 77.547 m | -44.3% |
| 最大单次 DVL 位置跳变 | 6.601 m | 0.000 m | 已消除 |
| 最大单次 DVL 全局航向分量 | 1.054° | 3.9e-14° | 数值零 |
| 终点误差 | 66.760 m | 77.184 m | +15.6% |

终点单点上未约束方案数值更小，但它在中途达到 139.2 m 后又偶然回落，因此不能用终点单值替代整段 RMSE、最大误差和跳变统计。该 A/B 也不是完整 SLAM 精度结论。

## Isaac Sim + 声呐前端 + GTSAM/TRN 全链路（2 km）

- 起点：30 km；真值初始化；5 Hz 同源多波束测深/侧扫，IMU 50 Hz，DVL 5 Hz，深度计 10 Hz。
- 运行状态：正常完成请求路段；共处理 6667 ping。
- Warp CUDA 声呐后端启用。

| 轨迹 | 3D RMSE | P95 | 最大误差 | 终点误差 |
| --- | ---: | ---: | ---: | ---: |
| 纯航位推算 | 14.233 m | 24.880 m | 25.722 m | 25.546 m |
| local 声呐 ESIKF | 5.136 m | 10.818 m | 14.216 m | 14.216 m |
| online global | 2.004 m | 3.761 m | 4.231 m | 4.012 m |
| 最终 GTSAM 优化 | 1.303 m | 3.095 m | 4.012 m | 4.012 m |

- local 相对纯航位推算的 RMSE 降低 63.9%。
- 与修改前同种子运行在 1.75 km 检查点比较，local 由 19.983 m 降到 6.106 m，下降约 69.4%。
- local 的相邻 5 Hz 误差样本最大变化为 0.0434 m，未再出现数米级离散跳变。
- local 航向 RMSE 为 1.135°、终点误差为 2.291°。最后 250 m 的位置误差连续增长到 14.216 m，说明剩余问题主要是弱地形段的声呐航向/沿航向可观性，而不是 DVL 瞬时拉偏；本次修正明显改善但没有消除长距离漂移。

## 软件验证

- 全部单元测试：346 passed。
- 新增三项针对性测试：DVL 规范子空间、DVL 继续修正速度但不回拉局部地图位姿、Joseph 后验协方差 reset 顺序。
- Python 编译检查、独立回归脚本直接启动检查和 `git diff --check` 均通过。
- pytest 在当前 shell 中需设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，否则 ROS 2 安装的外部 `launch_testing` 插件会因缺少 `lark` 在收集项目前失败；这与 OceanSim 代码无关。

## 方法边界与参考

当前实现仍是 OceanSim 自研的声学 FAST-LIVO2 风格 ESIKF，并不是官方 FAST-LIVO2，也不是完整的水下 InEKF 移植。本次采用的是可观测性约束的局部修正；完整 SE_2(3) 右不变滤波原型在同一导航回放中整体误差更差，所以没有直接替换现有前端。

- Potokar, Norman, Mangelson, *Invariant Extended Kalman Filtering for Underwater Navigation*: https://robots.et.byu.edu/jmangelson/pubs/2021/Potokar2021ral.pdf
- Solà, *Quaternion kinematics for the error-state Kalman filter*: https://arxiv.org/abs/1711.02508
- Zheng et al., *FAST-LIVO2*: https://arxiv.org/abs/2408.14035
- FAST-LIVO2 官方实现：https://github.com/hku-mars/FAST-LIVO2

## 复现文件

- 确定性 A/B：`standalone/dvl_observability_regression.py`
- A/B 数据：`demo/large_scale_slam/dvl_observability_regression_30km_4p5km_20260914.json`
- 全链路指标：`demo/large_scale_slam/dvl_oc_frontend_start30km_2km_20260914.json`
- 同步误差序列：`demo/large_scale_slam/dvl_oc_frontend_start30km_2km_20260914_position_errors.csv`
