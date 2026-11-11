# 三场景硬件在环数据字典

## 一、统计对象与坐标

四通道依次为truth、DR、Local、Global。truth是上位机仿真真值；其余为实际板端在线输出。位置单位为米，时间单位为秒，误差为同一时刻位置差的三维欧氏范数。未作事后刚体对齐、尺度拟合、轨迹平滑或坏样本删除。场景内坐标系一致，但三个场景并不共享一个地理坐标系。

Global是实际在线发布的位置，不是原始图中每个历史关键帧在试验结束后重新优化的位置。地图匹配、回环和后端输出链路见报告与各自部署源码；不可将最终历史图轨迹的误差替代在线误差。

## 二、完整轨迹格式

### 1. 海底和Chelan CSV.GZ

解压后为UTF-8带表头CSV，行数分别243569和157056。

| 字段 | 含义 |
|---|---|
| timestamp_s | 该声呐评价历元的仿真时间戳，不是墙钟处理耗时 |
| distance_m | 真值采样轨迹的累计三维弦长 |
| truth_x_m / truth_y_m / truth_z_m | 真值位置 |
| dr_x_m / dr_y_m / dr_z_m | 板端纯航位推算位置 |
| local_x_m / local_y_m / local_z_m | 板端局部里程计位置 |
| global_x_m / global_y_m / global_z_m | 板端在线全局输出位置 |
| dr_error_m / local_error_m / global_error_m | 对应板端输出的三维位置误差，正式统计对象 |
| native_dr_error_m / native_local_error_m / native_global_error_m | 同场景PC参考实现误差，用于原跨平台精度检查，不是板端主结果 |
| roundtrip_s | 当时记录的传输与处理往返耗时，不等同于纯算法耗时 |
| arrival_lateness_s | 结果到达相对原时间计划的累计迟到量，不等同于单帧计算耗时 |

这些位置CSV不包含完整姿态轨迹。报告使用的10/30/60秒SE3相对误差来自已有全姿态流独立审查，见relative_motion_audit.json；该审查结果和来源指纹已公开，但不能声称仅靠这些位置CSV重新计算了旋转RPE。

### 2. 千岛湖NX NPZ

通过numpy.load(path, allow_pickle=False)读取。

| 键 | 形状 | 含义 |
|---|---|---|
| timestamps | (62458,) | 仿真时间戳，秒 |
| distance_m | (62458,) | 原恒速路线参数，d=(t-t0)×0.8 m/s |
| positions | (62458, 4, 3) | 轴1依次truth、DR、Local、Global，轴2为x/y/z |
| error_m | (62458, 3) | 依次DR、Local、Global三维位置误差 |

最后样本航程16488.6441879272 m；要求的名义航程16488.7430927293 m，两者不能混写为相同数值。离散真值位置弦长总和16488.6365548619 m，与平滑路线参数相差约0.007633 m，由转弯离散采样引起，不据此修改原始航程字段。

本次虚拟路线垂向为常值，不构成任意六自由度贴底运动验证。记录间隔约0.33 s，对应本次约3.03 Hz的导航声呐输入；不能以早期5 Hz配置代替实际记录频率。

## 三、完整误差与指标

三份*_complete_error_samples.csv.gz均有相同表头：timestamp_s、distance_m、DR_error_m、Local_error_m、Global_error_m；保留每个同步评价样本。误差定义为e_i=||p_est,i-p_truth,i||₂，RMSE=sqrt(mean(e_i²))；P95为numpy.percentile(e,95)默认线性插值；末点是最后一个样本，不是最后一次进度打印。

verified_metrics.json中的metrics包含RMSE、median、P95、P99、max、final，单位米；segments为累计航程十等分区段，保留全部样本；tail包含误差超过5/10/20 m的样本占比、连续段数量和最长连续段样本时间跨度。

连续段跨度采用最后超限样本时间减首次超限样本时间。单个超限样本跨度为0，不人为添加一个采样周期。该统计不证明两个采样时刻之间的连续轨迹始终超限。

## 四、完整性与性能解释

源文件SHA256、板端与回传输出逐样本一致性、评价数组一致性和部署状态见evidence。公开的JSON副本去除了本机绝对路径、私有网络地址或凭据字段；对应原件与公开副本哈希分别记录，二者不同不表示数值修改。

海底源进程OOM、相对精度退化与实时积压必须保留。Chelan的完整性通过不等于硬实时通过。千岛湖receiver_summary中的estimator_wall_s是末次回调值，不能标成全程均值。模型全16路去噪耗时与导航实时链路使用不同计时边界。

## 五、资料边界

此包保存完整派生导航结果，不保存完整原始传感器输入。声学所原始MAT、逐ping幅度、复数信号、参考和查询强度场、网络凭据、模型权重、TensorRT引擎均未上传。algorithm_snapshot仅供检查实际算法，不保证脱离私有输入和原部署依赖即可复跑整场实验。
