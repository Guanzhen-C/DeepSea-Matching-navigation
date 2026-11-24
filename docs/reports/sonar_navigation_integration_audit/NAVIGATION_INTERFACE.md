# Pure Side-Scan Matching Navigation：从包到图优化

导航以 F 中冻结版本为准，F 与 R/H0/H5 的摘要关系见[数据一致性](DATA_CONSISTENCY.md)。本页标注 `F/...` 的文件不可用当前工作区同名实现代替。

## 1. 实际输入：不是声呐图像，也不是神经网络的增益

**已验证**：`F/standalone/PureSideScanFeatures.py:14–42` 的 `SideScanFeaturePacket` 含 `timestamp`、同长度 1D 的 `ranges_m/amplitudes/rows/sides/sample_sequence`、8 个递增的 `row_centres_m` 和 `max_range_m`。除有效返回外不添假点；对空/坏支持有后续检查。幅度和范围在包内复制为 `float64`，束/舷/索引为 `int64`，观测数组只读。该版本不接收 log gain、复相位或裸 PNG。**已对准标定的数值幅度可以用**：通过接收适配器构造完整包，不能仅传一个强度矩阵。

| 构造环节 | 真正执行的步骤 |
|---|---|
| R：`RunQiandaoRecordedLegacyNavAB.received_packet:44–59` | 两臂共用候选支持 $w>0.5\land z\ne0$。按束用 `INTER_AREA` 将带支持 $A^2$ 汇聚至 512 格取平方根；支持占比 >0.6，另按各臂全 ping 最大幅度的 $10^{-6}$ (带数值下界) 剔除接近零样点。$R_{\max}=N c/(2 f_s)$、$c=1480$、$f_s=37650$，第 $j$ 格 $r=(j+0.5)R_{\max}/512$，行中心 $(b-3.5)\times0.15$ m。这里的 $j$ 是导航接收格，不是原生样点 $k$。 |
| H0/H5：`SideScanIQHistory.packet_from_compressed_IQ:69–88` | 同一接收脉压之后，对 $|z|^2$ 做按束 512 格面积汇聚，再开方；检测门限 $\max(A_{\max}10^{-6},\mathrm{tiny},1.8\sigma_{\rm thermal})$，两臂接收器参数相同但通过的格不强制相同。$R_{\max}=250$ m，斜距中心按相同格规则；行中心含接收器前向 +0.10 m 外参。没有把原生恢复的水柱 gate 当作本函数的检测掩码。 |
| 运动输入 | `F/standalone/PureSideScanFeatureNavigation.py:134–142,205–217`：`imu(stamp,gyro,accel)`、`dvl(velocity)`、`depth(depth_m)` 分别驱动两条 DR/Local ESIKF；在声呐时刻补用保持的 IMU 做传播。DR 只用运动流，Local 才吸收有效侧扫前端更新。 |

## 2. 特征构造与相对前端

`F/standalone/PureSideScanFeatures.py:88–151`：`image_from_packet()` 将同舷样点按物理斜距放到 8×1024 低级栅格，左舷列翻向中心、右舷顺序展开，重复格取最大幅度。随后 log1p 压缩、按距离列的中值增益插值和高斯平滑、2–98 分位归一、沿束行 4 倍三次插值（掩码近邻）、CLAHE。GFTT 角点上限 900，以 `cornerSubPix` 精化后计算 L2 归一 SIFT；不是 P32 那个 RootSIFT 检测器。插值后沿程像素中心以 0.15 m 行距逆变换；横轴以网格比例写有符号斜距，不推断地标 XYZ。

`descriptor_pairs/robust_feature_row_motion:154–207`：L2 近邻双向 ratio=0.9、同舷约束；至少 24 对及至少 24 内点，按 ping 时间种子做 256 次小样本一致性，物理行残差阈值 0.15 m。模型拟合沿航向位移与斜距方向的外观剪切；剪切不是被测航向，只有沿航向量更新 Local。函数对位移修正软缩放；`F/standalone/PureSideScanFeatureNavigation.frontend:161–203` 的三次连续增量形成四帧链后才对锚定方向做一维更新；不足则保留惯性/DVL/深度并记录拒绝。**结构有效不等于幅度鲁棒**：归一化、检测、描述子、ratio 及门限全可能随去噪变化。

## 3. 先验强度图、滑窗子图和两路绝对匹配

- **先验图构建和加载是两件事。** R 的 `PrepareQiandaoRecordedTrackPrior.main:73–142` 以偶序号回波、冻结的 56M 离线增益、实测接收角/时延与 GNSS 姿态代理投影真实回波，栅格均值后在 1.5 m 距离预算内有限插补，再写 `backscatter.npy`、mask、metadata，分辨率 0.3 m。H0/H5 使用各自冻结的参考图，非 7.52M 在线生成。仿真宿主生成场有几何，板端 `F/standalone/SideScanFeatureMap.IntensityFeatureMap.from_path:40–52` 只载强度、支持掩码、分辨率，不加载高程。`sample:54–68` 仅在四邻皆有效时双线性采样，不在先验掩码外造观测。
- **条带子图**：`SideScanFeatureWindow.build:92–188` 使用最近 150 个导航 ping、估计姿态与束相位中心，把扫描平面投到常规条带（沿程步长 0.5 m）；普通重叠均值，不做第二个去噪器。子图重采样 `PureSideScanFeatures.resample_supported_echoes:62–85` 只用相邻采集索引的夹持样点线性插值，不跨洞、不外推；通常以每舷 512 格保持物理斜距刻度。启用来源时保留一个主贡献的相位中心和距离列，并保存包 ID、时间和束表；同一栅格中强度最大者决定标签，不证明就是最可靠地物。
- **条带绝对匹配**：`SideScanFeatureMap.match_feature_prior:191–315` 在估计种子附近（沿程 ±30 m；最终解约束见代码）采样先验强度；本轮用 LoFTR 做外观对应，并检查同舷，24 对门槛后做扫描平面一致性。默认条带 `rank2_baseline` 仅沿程和航向约束，横向保留运动种子，不能把有限的粒子 proposal 协方差当额外测量信息。
- **转弯投影与绝对匹配**：`PureSideScanFeatureNavigator.side_scan:218–246` 当 `adaptive_projected` 且有历史航向与末帧航向的余弦 <0.5 时，用 `SideScanProjectedSubmap.build_projected_submap:44–119`，以估计位置/姿态和 slant proxy 排列**外观**，保留真实来源的相位中心/方向/距离列。 `SideScanProjectedMatching.tiled_feature_matches/match_projected_prior:16–90`：256×384、32 格重叠的 LoFTR，去重后取先验 XY、源扫描平面约束和可观子空间；投影像素 XY 不当水平实测。LoFTR 用外部公开户外模型，未在本试验重新训练；`SideScanLearnedFeatures.py:22–68` 默认 mask 不进网络上下文，但输出点按两侧实际掩码过滤。

## 4. 历史回环与成对斜距是不同约束

**历史相对回环**（`SideScanLoopClosure.py:14–25,64–208`）：候选要求时间分离、不重用重叠 ping 窗、估计几何支持；按强度描述子相似性与距离选最多 3 个旧窗。普通条带可按航向反向翻转外观后 LoFTR；含投影的混合子图调用 `SideScanHistoricalMatching.match_typed_historical:38–84`，用相对估计作 tile 仿射对齐、线性插值强度/近邻掩码，坐标逆变换回源像素。图像仿射 RANSAC 只是 `appearance_consensus:87–95` 的外观筛选，不能当导航位姿。最终进入 `SideScanLoopGeometry.fit_range_plane_loop`：联合两次扫描的斜距/扫描平面，估计未知公共地标与相对位姿；需时序分离窗确认才插入回环边。条带回环采用插值行的平面/起点和子像素列斜距；投影回环走主贡献来源。H0 对非正等无效斜距的候选先行拒绝，不修改几何求解器（`HistoricalLoopObservationGuard.py:12–60`）。

**与地图关联的重复斜距**（`SideScanMappedRangeBank.py:18–41,44–54,66–133`）：只在确认的地图窗口，从投影标签或条带来源表取 $(\mathrm{packet},\mathrm{beam},\mathrm{signed\ range\ column})$、源相位中心与斜距中心；其有符号列同时含舷侧。按约 1 m 地图 XY 关联、至少 180 s 间隔、独立不交叠窗、至少 6 特征/因子，剔除同窗近并列候选；最多取 48 对。去重复使用 token 是保守策略，不等于消除与地图约束的全部相关性。压力辅助 z 是载体估计量，在 R 为名义深度相关估计，不是已知地标高程。

对地标水平坐标 $L$，由两端位姿确定相位中心 XY $O_a,O_b$，斜距 $r_a,r_b$，计算

$$
d=\|L-O_b\|^2-\|L-O_a\|^2-r_b^2+r_a^2,\qquad
e=\frac{d-\operatorname{clip}(d,l,h)}{(r_a+r_b)\sigma}.
$$

$[l,h]$ 由两次估计的换能器高度差与允许的目标未知高程区间导出；深度相等时退化为等式。 `SideScanMappedRangeChange.profiled_range_residual_jacobians:93–135` 再做 pseudo-Huber、信息预算（默认 0.5、独立等效特征 24）并给出解析雅可比；`conservative_range_change_sigma:138–152` 以测距和估计基线传播不确定性。未读入 DEM/真值高程，区间内零残差不凭空增加约束。

## 5. GTSAM 处理什么，输出什么

**已验证**日志 `diagnostics.optimizer_backend=gtsam_isam2_incremental_fixed_lag`。图的变量是关键帧平面 $x,y,\psi$（`Pose2`），不是原生回波或神经网络特征。 `PureSideScanFeatureNavigation.py:23–32,72–88,148–159,257–315` 给图：

1. 初始 gauge 与局部里程计相对边（常规关键帧 30 s，确认证据可强制建帧）。
2. 粒子组跨独立窗口确认的绝对侧扫先验（兼容旧 API 的 `terrain_prior` 类型后重标为 `side_scan_feature_prior`，不是高程）。同一当前窗口用于绝对和回环时各按一半信息计入，近似处理共享证据。
3. 确认的历史回环相对边；另有可选 `MappedRangePoseGraph.add_mapped_range_change/_edge_factor:35–61`，挂到两个实际 Pose2：零 SE(2) 只是表项占位，因子是真正的 `gtsam.CustomFactor`，以解析世界雅可比换为 Pose2 局部切空间，误差已经按 $\sigma$ 白化，噪声模型单位尺度。现用 `preserve_legacy` 不全局缩减原先验/回环精度，但前述共享窗分账仍在。

`F/core/oceansim/slam/pose_graph.py:736–750,813–872,994–1068,1096–1146,1235–1263` 分别维护全历史 iSAM2 和 180 s 增量固定滞后图；碰到已边缘化端点的远程因子只进全历史图，并经刚性全局锚反馈到活动窗口（`_bridge_global_anchor_to_fixed_lag_window:953–993`）。输出优化平面位姿；导航把最新图关键帧相对局部的校正 `raw_graph_pose()` 经 `MapOdomSoftCorrector.update()` 发布 Global 的水平位置和航向，Global z 仍来自 Local；不将后端位姿直接写回 ESIKF。**R 已运行，但历史回环及 mapped-range 实际插入均为 0**，不能用 R 的误差结果验证这两类因子；H0/H5 两类均有真实插入，见[实际流程](END_TO_END_FLOW.md)。
