# 两时刻位姿雅可比

状态：世界坐标解析式和局部坐标转换**已证明**；分段光滑域**已数值核验**；非退化区间端点的经典残差导数不存在。[实际运算](../../../standalone/SideScanMappedRangeChange.py#L83) · [GTSAM 包装](../../../standalone/SideScanMappedRangeGraph.py#L45) · [图的局部增量变换](../../../isaacsim/oceansim/slam/pose_graph.py#L772)。

## 1. 按代码链式推导

令 $K=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$，$d_i=R_iKu_i$。代码以行向量实现：偏移乘 $R_i^\top$，偏航导数乘 $K^\top R_i^\top$。$do_i=dt_i+d_i\,d\psi_i$，所以 $d\|v_i\|^2=-2v_i^\top do_i$。

$$J_{D,a}=[2v_a^\top,\ 2v_a^\top d_a],\qquad J_{D,b}=[-2v_b^\top,\ -2v_b^\top d_b].$$

令 $\chi=1$ 当 $D<\ell$ 或 $D>h$，以及当 $\ell=h$；否则源码设 $\chi=0$。在非端点处这是 $\partial q/\partial D$；点区间对应 $q=D-\ell$，处处导数为 1。范围、压力高度、冻结的行标准差与索引固定，因此 $\partial e/\partial p_i=\chi J_{D,i}/(B\sigma)$，其中 $e=y/\sigma$。没有对图节点快照中估计的基线求导，也没有对选出的关联或区间端点求位姿导数。偏航只有通过杆臂和节点到该 ping 的水平位移起作用。

代码以 $s=\sqrt{1+(e/\delta)^2}$ 定义 $\phi(e)=e\sqrt{2/(s+1)}$，于是 $\phi^2=2\delta^2(s-1)$。由 $s'=(e/\delta^2)/s$ 及 $e^2/\delta^2=s^2-1$，

$$\phi'(e)=\sqrt{(s+1)/2}/s.$$

最终 $E=w\phi(e)$，$J_{E,i}=w\phi'(e)\chi J_{D,i}/(B\sigma)$，其中 $w=\sqrt{\beta\min(n,n_0)/n}$。关闭鲁棒时 $\phi(e)=e$。残差无量纲、平移列单位 m⁻¹、偏航列 rad⁻¹。对应[截断与雅可比，第 115–124 行](../../../standalone/SideScanMappedRangeChange.py#L115)，[鲁棒与预算，第 125–135 行](../../../standalone/SideScanMappedRangeChange.py#L125)。

GTSAM 的 Pose2 本地增量在零点满足 $dt_i=R_i\,d\xi_{i,xy}$、$d\psi_i=d\xi_{i,\theta}$，故回调传入 $J_{E,i}\operatorname{diag}(R_i,1)$，而非世界坐标 Jacobian；位姿切空间的正确比较是 Pose2.retract。CustomFactor 使用单位噪声，并无第二层鲁棒白化。该因子的 GTSAM error 为 $\tfrac12 E^\top E$；内部诊断代价使用 $E^\top E$，比较日志时要统一约定。[回调](../../../standalone/SideScanMappedRangeGraph.py#L45) · [诊断代价](../../../standalone/SideScanMappedRangeGraph.py#L63)。

## 2. 截断端点

非零区间的端点，残差的内侧导数为 0，外侧为上式，非切向扰动下不存在两侧统一导数。代码选择内侧值。在端点 $E=0$，平方或伪 Huber 代价的两侧一阶导数均为零，二阶曲率仍有跳变。中心差分跨过端点通常得到半个外侧方向导数，不能当作符号错误。实验应测两侧单边差分，并把它与完全等高时的可微点区间分开。

## 3. 数值证据

项目测试包含[世界坐标中心差分](../../../tests/test_sidescan_mapped_range_change.py#L44)、[区间／共基准](../../../tests/test_sidescan_mapped_range_change.py#L60)、[GTSAM 切空间差分](../../../tests/test_sidescan_mapped_range_graph.py#L46)和[全局图保留长边](../../../tests/test_sidescan_mapped_range_graph.py#L84)。本轮独立检查另在冻结运行时上对比世界坐标无/有鲁棒残差及 GTSAM CustomFactor 的局部线性化，参数、样本和结果见[实现审查](IMPLEMENTATION_REVIEW.md)。数值吻合仅支持被执行分支的导数，不证明关联、噪声标定或线性化收敛域。
