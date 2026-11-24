# 因子相关性、预算与 token 去重

## 1. 三种信息的实际来源

| 因子 | 实现与来源 | 潜在重叠 |
|---|---|---|
| 地图匹配 | [地图定位接口](../../../standalone/SideScanFeatureMap.py#L191)与[导航调用](../../../standalone/PureSideScanFeatureNavigation.py#L218)：当前侧扫子图与有地理参考的强度图匹配，产生地图位姿观测和逐特征地图 XY 及来源 | 与重复斜距共用当前子图及接受匹配；地图强度与地图几何带共同地理误差 |
| 历史回环 | [历史匹配](../../../standalone/SideScanLoopClosure.py#L118)、[距离/平面](../../../standalone/SideScanLoopGeometry.py#L25)：用两时刻各自斜距、扫掠平面、侧向支持和 3D 地物求相对信息 | 可能用同一对/相邻 ping、同一地物或同一局部里程计；无跨因子 token 排他或联合协方差 |
| 重复斜距 | [Bank 配对](../../../standalone/SideScanMappedRangeBank.py#L66)：以通过地图匹配的记录生成依赖双节点的斜距变化因子，两次相位中心相对节点的偏移冻结 | 不是与地图匹配独立的新传感器；即使不共用某标准列也可能共用轨迹与底图 |

历史回环的纯数据 Schur 信息去除局部求解辅助弱先验，不会消掉跨块相关性。[回环信息求法](../../../standalone/SideScanLoopGeometry.py#L52)。里程计虽然不按同一预算缩放，重投影偏移依赖局部姿态轨迹，其相关性也不能忽略。

冻结回环的计算可以明确写出。对在旧窗口坐标下的地物 $P=(P_{xy},Z)$，由待估相对平面位姿 $(t,\psi)$ 和已知窗口垂向差 $\Delta z_w$ 得到新窗口地物坐标 $P_b=(R^\top(P_{xy}-t),\,Z-\Delta z_w)$；旧窗口 $P_a=P$。令 $d_i=P_i-o_i^{3D}$，每视图保留两行 $(\|d_i\|-r_i)/\sigma_r$ 和 $n_i^\top d_{i,xy}/\sigma_\pi$，$n_i$ 为扫描平面水平法向。舷侧筛查为 $\mathrm{side}_i(Kn_i)^\top d_{i,xy}\ge-\mathrm{margin}$。这是与斜距变化约束不同的四行观测模型，偶尔却来自同一次测距。参见冻结版的[距离/平面解析导数](../../../standalone/SideScanLoopGeometry.py#L25)及[舷侧/拟合](../../../standalone/SideScanLoopGeometry.py#L77)；冻结 F 中对应第 25–50、77–85、87–176 行。

冻结回环对每个地物 $j$ 用数据 Jacobian $[J_{p,j},J_{\ell,j}]$ 消去三维地物，再累积 $\sum_j J_{p,j}^{\top}(I-J_{\ell,j}J_{\ell,j}^{\dagger})J_{p,j}$；这里两组 Jacobian 已带软鲁棒行尺度。输出对称化、谱阈值保留秩、按特征数降权并加模型方差，不包含用于局部求解的弱位姿种子行（F 第 52–74、123–134、161–175 行；[冻结版源码](../../../standalone/SideScanLoopGeometry.py#L52)）。最小秩筛选及物理侧检是**回环**策略，重复斜距边没有同等筛选。

## 2. Bank 的去重范围

- [第 18–41 行](../../../standalone/SideScanMappedRangeBank.py#L18)：token 为（packet ID、波束行、原图列）；从来源列按所属舷侧解码距离。来源是栅格主贡献者，不是未重采样的 ADC 成分。不同 token 并不表示不同足迹或不同物理地物。
- [第 66–109 行](../../../standalone/SideScanMappedRangeBank.py#L66)：同一 Bank 的窗口时间严格向前；当前 token 用全局已见集合登记，旧 token 只能取未用的，临时预留避免本轮重复；要求历史节点更老、时间和地图距离合格，禁止窗口重叠。当前 token 以前未见，故在 Bank 身份语义正确、状态不重置时，同一个精确 token 不会成为两条已提交斜距边的输入。成边后按实际配对登记已用，[第 110–133 行](../../../standalone/SideScanMappedRangeBank.py#L110)。
- 保证仅限 Bank 内的确切 ID 和状态。没有把校准版本/航次编码入 token；重启、跨源拼接或复用 packet ID 需命名空间。地物无全局唯一 ID；邻近列和不同 ping 仍可能采同一散射物。诊断“没有复用原始斜距样本”的常量不是独立性检验；三次以上观测在不同边里也不会显式共享同一个高度。
- [第 90–141 行](../../../standalone/SideScanMappedRangeBank.py#L90)：默认 1 m 地图邻近、至少 180 s，相邻窗口不能重叠。若排名最靠前的两项在同一旧窗口且距离过于接近，才过滤该近邻歧义，不是跨窗口全局唯一性。按旧节点成边，至少 6 特征，最多取 48 并按距离分散，未按 Jacobian 条件数或外点概率选样。候选可能在成边前即预留，首次见到的 token 也可能未使用就不再重试；存在召回损失。分桶限存、间隔与距离分散都不能证明独立。
- 最小间隔比较的是单个来源 ping 时间，不是图节点相隔时间。它接近既有固定滞后设置；应记录因子是进活动窗还是只进全局图，不能只由 180 s 推断激活机制。按地图位置邻近避免跨远处配对，但会随地图误差、航线间距和重复纹理变化；宜按航次在有真实重复点标注时校准半径及时间限制，并报告丢失的正确配对。

## 3. 信息预算与伪 Huber

重复因子自身 $w^2=\beta\min(n,n_0)/n$，通常 $\beta=0.5,n_0=24$；单边行系数和不超过 $\beta n_0=12$。这不是 Fisher 信息矩阵、谱范数、全图总信息或真实独立样本数的上界；多个相关边仍可能过置信。伪 Huber $\delta=3$ 是对标准化残差的软抑制；只有白化标定正确才可解释为“约三倍标准差”的尺度，也不作硬拒绝。[默认](../../../standalone/SideScanMappedRangeChange.py#L40) · [实现](../../../standalone/SideScanMappedRangeChange.py#L125)。

[Graph 第 18–33 行](../../../standalone/SideScanMappedRangeGraph.py#L18)的 legacy precision budget 只缩放已有地图/回环信息，不缩放里程计。冻结版导航在当前窗口同时产生地图和历史回环时分配共享精度（[导航调度](../../../standalone/PureSideScanFeatureNavigation.py#L286)、[回环分配](../../../standalone/SideScanLoopClosure.py#L183)）；重建历史实验须读配置。preserve_legacy 与 tempered 不只是改变斜距开关。

**结论：**可以写“精确 token 不重复”“对可能相关的观测做工程降权”，不能写“三类因子独立”“信息预算已证明统计一致”。建议记录各因子的子图、token 集合、地图分块和轨迹锚点，建立交叠矩阵；先用固定关联重放比较排他选择、协方差膨胀和联合模型，再验证独立航次的置信覆盖率。如果引入共同地物变量，应保留跨行协方差或在残差层统一白化。
