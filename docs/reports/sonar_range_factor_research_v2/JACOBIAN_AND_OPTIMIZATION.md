# 两时刻位姿导数、Schur 与 GTSAM

以 $F=\tfrac12 q^\top Wq$ 定义，原型只实现 Pose2 的 $x,y,\psi$；垂向估计不被该因子直接优化。令 $v_i=L-o_i,\ u'_i=R_i u_i,\ K=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$，水平导数块 $G_i=[I_2,\ Ku'_i]$。从 $\rho_i^2=v_i^\top v_i+(Z-z_i)^2$ 得

$$a_i=\partial_{p_i}\rho_i=-v_i^\top G_i/\rho_i,\qquad b_i=\partial_Z\rho_i=(Z-z_i)/\rho_i.$$

只填对应位姿的列；同一位姿节点在多次观测中出现时须按链式法则相加。对各观测预测，二阶块为

$$\rho_{p_i p_i}=(G_i^\top G_i-a_i^\top a_i)/\rho_i+
\left(v_i^\top u'_i/\rho_i\right)e_\psi e_\psi^\top,\quad
\rho_{p_i Z}=-a_i^\top b_i/\rho_i,\quad
\rho_{ZZ}=\|v_i\|^2/\rho_i^3.$$

第一项来自对平方距离微分，偏航曲率附加项来自 $d^2v_i/d\psi_i^2=u'_i$。取联合未白化 Jacobian $D=[A,b]$，则

$$\nabla_{p,Z}F=D^\top Wq,\qquad H=D^\top W D+\sum_i(Wq)_i\,\nabla^2\rho_i.$$

如果 $W$ 随优化位姿一起变化，需要协方差导数及似然归一项；原型明确冻结 $W$。

## 剖面目标及光滑性

唯一稳定内部最优 $Z^*$ 且 $H_{ZZ}>0$ 时，隐函数导出 $Z_p^*=-H_{Zp}/H_{ZZ}$。固定域的 Envelope Theorem 给

$$\nabla f= A^\top Wq,\qquad \nabla^2 f=H_{pp}-H_{pZ}H_{ZZ}^{-1}H_{Zp}.$$

严格活跃的固定端点 $Z_p^*=0$，此时 Hessian 是 $H_{pp}$。特殊等高情况，所有接收器恰在固定上界，$F_Z(Z_+)\equiv0$；若 $H_{ZZ}>0$、该上界为唯一全局极小值，它仍稳定而非切换，使用同一边界表达。真正的活跃集切换处，唯一最优仍可给一阶 envelope 梯度（固定紧域和连续导数），但一般只有分段二阶；并列最优若 $\partial_pF$ 不同，不能宣称值函数可微。参见[Milgrom–Segal 的包络定理原文](https://web.stanford.edu/~milgrom/publishedarticles/Envelope%20Theorems.pdf)及[实现](../../../research/sonar_range_factor_v2/models.py#L222)。临界点不硬用零导数或伪 Hessian。

## 残差与原优化的一致性

令 $e=Cq(p,Z^*)$，$C^\top C=W$。用**带符号向量**（非 $\sqrt{2f}$）输入最小二乘，半平方代价恰为 $f$，光滑分支上 $J_e=C(A+bZ_p^*)$。零代价附近仍可保留符号和非零雅可比；标量 $\sqrt{2f}=\|e\|$ 则一般在真解有绝对值型折角。光滑分支中 $J_e^\top e=\nabla f$，但 $J_e^\top J_e$ 是 GN 近似，非完整剖面 Hessian。

对固定工作点的线性最小二乘 $\min_{\delta Z}\|e+J_p\delta p+B\delta Z\|^2$，正交投影 $P_\perp=I-BB^\dagger$ 精确得到 $\|P_\perp(e+J_p\delta p)\|^2$；等价于对该线性系统做 Schur。它**不是**任意点精确非线性剖面残差的替代。对内点零残差，精确 Jacobian 化成 $P_\perp C A$；有非零残差时需保留曲率项。临界端点应投影到带符号可行锥，不能用自由 Schur；严格端点不消元高度增量。两视图里零高度 Jacobian 的情况不可用除法强行制造约束。[线性投影](../../../research/sonar_range_factor_v2/models.py)。

## GTSAM 原型验收

已实现[双键单位噪声 CustomFactor](../../../research/sonar_range_factor_v2/gtsam_adapter.py#L7)，在光滑工作点返回两行和局部 Pose2 Jacobian，使用 $J_{\rm world}\operatorname{diag}(R_i,1)$；单位噪声避免二次白化。除了世界加法有限差分，测试对 GTSAM 的 Pose2.retract 做中心差分，并放入带位姿先验的小图实际优化（[测试文件](../../../tests/test_sonar_range_factor_research_v2.py)）。符合[官方自定义因子接口](https://gtsam.org/tutorials/intro.html#defining-custom-factors)。没有证明 LM 或 iSAM2 在活跃集切换、并列最优、错误关联时稳定；目前回调在该类非光滑状态显式失败。正式接入前应采用可重试的分段信赖域、必要时局部显式有界高度或回退 M0，记录每次回退，并检验切换时的全图行为。
