# 实现审查和可核验的数值证据

## 1. 版本与调用关系

本轮读取工作区源码（基线提交 4a09ff1，工作区存在其他改动），并对照第二阶段审查的冻结运行时 F。三个核心文件在本轮读取版本与 F 字节一致；历史航次导航调度和回环不得用工作区新路径反推。[第二阶段版本与档案定位](../sonar_navigation_integration_audit/README.md) · [历史报告的独立运行时清单](../seabed_matching_improved_algorithm_20261107/evidence/report_runtime/runtime_manifest.json)。

| 核心文件 | 本轮与 F 共同 SHA-256 |
|---|---|
| SideScanMappedRangeChange.py | a5f3640b954547669aa5ce01f3c2ea34110267e7baee2b16f4568006ca3bef42 |
| SideScanMappedRangeBank.py | 906575f81b92605382b52de5a1d7963f5f2790847608cde21dce1770d11db460 |
| SideScanMappedRangeGraph.py | c577a25bc684195e4f7c4fc8a9eb3d25e8b6a8ecbec9344d886005feb9b5a2c |

F 位于本机论文资料库的“02_匹配导航算法/复现实验_冻结导航运行时/runtime”。F 的 nav / map / loop 几何与当前同名文件不同。F 行号索引：导航 218–246 选择条带/投影及匹配，257–284 产生并确认绝对证据，285–310 分配同窗精度并尝试插边，311–315 返回输出；地图 119–160/183–187 生成来源，191–315 构造匹配；回环几何 25–50 实现两时刻单独的距离/扫掠平面残差和雅可比，52–67 取数据 Schur 信息，77–85 检查舷侧，87–176 求解。导航、地图、投影、回环调度及几何这五个文件的 F 版本与 GitHub 基线 4a09ff1 字节一致。因此网页链接指向[冻结导航第 218 行](../../../standalone/PureSideScanFeatureNavigation.py#L218)、[插边第 305 行](../../../standalone/PureSideScanFeatureNavigation.py#L305)、[地图来源第 119 行](../../../standalone/SideScanFeatureMap.py#L119)与[回环第 25 行](../../../standalone/SideScanLoopGeometry.py#L25)；本机未提交工作区另有变体，插边入口及回环 Schur 分别位于当地第 436 与 136 行，不作为已发布证据。实采和板卡的各版本结论以第二阶段和原始回执为准。

## 2. 公式—代码核对

| 项目 | 具体源码 | 审查结论 |
|---|---|---|
| 有限数组、支持范围和冻结观测 | [Change 30–68](../../../standalone/SideScanMappedRangeChange.py#L30) | (N,2) 的地图 XY / 局部 XY，(N,) 的 Z、斜距、标准差；有限、正斜距、不超量程；还未验证物理双距一致性 |
| 量程列和实际接收偏移 | [Bank 18–41](../../../standalone/SideScanMappedRangeBank.py#L18)、[Bank 66–89](../../../standalone/SideScanMappedRangeBank.py#L66)、[投影来源](../../../standalone/SideScanProjectedSubmap.py#L29) | 只接真实接收来源标签；条带的来源像素使用近邻索引，投影路径取贡献者斜距；$u$ 是从局部轨迹重投影的固定偏移，$z$ 为辅助导航估计 |
| 因子条件与冻结地图坐标 | [Bank 90–141](../../../standalone/SideScanMappedRangeBank.py#L90)、[地图定位接口](../../../standalone/SideScanFeatureMap.py#L191) | 过去节点、≥180 s、窗口不重叠、地图 XY 距离 ≤1 m；成对 XY 平均；至少 6、最多 48。地图对应从以种子位姿渲染的固定先验条带解算，不是真实地物独立量测。不保证同一真实散射点、不按可观秩选样 |
| 刚体原点、平方差、高度投影 | [Change 83–120](../../../standalone/SideScanMappedRangeChange.py#L83) | 与 [数学模型](MATHEMATICAL_MODEL.md) 一致；只剖面平方差，端点非光滑，死区无法区分部分错配 |
| 行尺度、鲁棒与雅可比 | [Change 119–152](../../../standalone/SideScanMappedRangeChange.py#L119) | 解析导数正确，使用冻结且部分近似的标量噪声；未传播跨观测相关和差分高度误差 |
| 插边与线性化 | [Graph 35–61](../../../standalone/SideScanMappedRangeGraph.py#L35) | 双键 CustomFactor，单位噪声，返回解析雅可比；零 SE2 边仅占位。有限差分是验证方法，不是生产求导 |
| 诊断与重放 | [Graph 63–84](../../../standalone/SideScanMappedRangeGraph.py#L63)、[历史输入审计](../../../standalone/AuditMappedRangeInputs.py) | 保存输入与策略；诊断平方代价和 GTSAM 半平方约定不同；组合实验需记录被拒候选、源窗和校准版本，不能用诊断常量替代证明 |
| 图优化与输出 | [pose_graph.py 772–780](../../../isaacsim/oceansim/slam/pose_graph.py#L772)、[953–993](../../../isaacsim/oceansim/slam/pose_graph.py#L953)、[994–1146](../../../isaacsim/oceansim/slam/pose_graph.py#L994) | 同时维护固定滞后与全历史；过期端点远程边进入全历史，活动窗口走锚反馈。Global 垂向仍是 local，并不由该因子估计；详见[上一阶段输入输出审计](../sonar_navigation_integration_audit/NAVIGATION_INTERFACE.md) |

未发现平滑分支上 $D$、解析导数、局部切空间变换的代数错误。已确认三个实现限制：把包络当严格物理可行域会错误接受不可行原距；“conservative”噪声函数不具备普适上界；同一次无异常的数值线性化不能证明可观性或概率一致性。无脉冲支持、非有限数据、非正/超量程能在入口被拒绝；时间顺序拒绝不能检出水声多径、地图错配、同源重复信息或模型外的校准偏差。冻结 H0 的历史回环还有[独立观测防护](../sonar_navigation_integration_audit/NAVIGATION_INTERFACE.md#L30)（原文件 HistoricalLoopObservationGuard.py 第 12–60 行），不能误记为重复斜距的增强。

## 3. 本轮验证（非新增导航航次）

- 58 项针对性测试通过：斜距数学、GTSAM、Bank、原始来源审计、独立不变量和回环几何。用当前 standalone 与冻结核心依赖执行；已有 tests 不覆盖所有高度死区、真实传感器误差或未知环境。
- 独立 Python 探针仅导入 F 的 standalone 和 core，NumPy seed 731、120 组×17 行；测试 $\Delta z\in\{0,\pm10^{-6},\pm0.25,\pm12\}$ m，随机位姿、偏移、多方位地物和相对高度。世界坐标直接加减 $10^{-6}$，局部坐标分别调用两个节点的 Pose2.retract；对每组验证实际 CustomFactor 的线性矩阵与误差。剔除距非退化截断端点不超过 $10^{-3}$ m² 的 187 行，剩 1,853 行（其中 1,111 激活）。对六列所有被检元素，世界无鲁棒、世界有鲁棒、GTSAM 局部的最大绝对差依次为 $8.93\times10^{-9},7.20\times10^{-9},6.75\times10^{-9}$；局部线性矩阵和 $J_{\mathrm{world}}\operatorname{diag}(R,1)$ 差不超过 $2.23\times10^{-16}$，半平方代价差不超过 $1.43\times10^{-14}$。此处是数值一致性，不是模型误差。
- 单独构造上界：$z_a=0,z_b=2,o_a=o_b=0,L=(1.5,0),r_a=1.5,r_b=2.5$，令 $\sigma_p=0,\sigma=w=1$。此时 $D=h=-4$ m²；扰动目标 $x$，程序解析选 0，实测左导数约 $-0.75000025$、右导数 0、中心差分约 $-0.37500012$。前文两条 5 m 原距均物理不可行的反例，程序残差确为 0。
- 合成锚定源节点 $(0,-5,0)$、目标 $(0,0,0)$，地物 $Z=-8$ m。两中心等高时，目标在同方位地物、异方位但无杆臂、异方位并有变杆臂时的秩分别 **1/2/3**；最后一例将目标接收器高度变为 $10^{-6}$ m 后重算真斜距，死区内部秩为 **0**。这不是航次的秩分布。

已有测试的复跑方法（变量指向第二阶段冻结运行时 core，而非板卡）：

~~~bash
audit_core='output/论文数据资料库_20261124/02_匹配导航算法/复现实验_冻结导航运行时/runtime/core'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$audit_core" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-sonar-denoise/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_sidescan_mapped_range_change.py tests/test_sidescan_mapped_range_graph.py \
  tests/test_sidescan_mapped_range_bank.py tests/test_mapped_range_input_audit.py \
  tests/test_mapped_range_independent_invariants.py tests/test_sidescan_loop_geometry.py
~~~

## 4. 既有导航实验，不与本轮数学验证混用

| 既有实验 | 只尽量保持旧信息的斜距开启对照：Global RMSE / 末端误差 | 新增因子 / 特征对 |
|---|---|---|
| [主仿真 seed 31，16.49 km](../seabed_matching_improved_algorithm_20261107/evidence/main_independent_comparison.json#L1) | 11.733→5.040 m / 7.928→9.940 m | 277 / 3119 |
| [另一次仿真 seed 33，4.00 km](../seabed_matching_improved_algorithm_20261107/evidence/fresh_independent_comparison.json#L1) | 7.629→5.951 m / 12.658→1.591 m | 19 / 202 |
| [开发仿真 seed 32，2.00 km](../seabed_matching_improved_algorithm_20261107/evidence/preserved_development.json#L1) | 7.797→7.874 m / 2.083→4.693 m | 8 / 110 |

主实验和 seed 33 的回执审计原始输入及事件，DR/Local 保持一致；主实验首条新因子前 Global 一致。但整个导航在线运行，斜距边改变后续图状态和可能的关联，故不能替代固定关联图重放。地图、路线等仍属同仿真环境系列，不能当独立海湖外场。tempered 同时修改 legacy 精度，非纯开关。第二阶段 [H0/H5、R 设备记录](../sonar_navigation_integration_audit/END_TO_END_FLOW.md) 中仿真 HIL 两臂均启用斜距与回环，实际插入只能证明链路执行；实采 R 两类新增因子均为 0，不能验证其现场定位增益。

设备回执（顺序为原始／去噪）：[H0](../sonar_navigation_integration_audit/evidence/paired_run_receipts.json#L24) 实际插入斜距 312／265、回环 322／314；[H5](../sonar_navigation_integration_audit/evidence/paired_run_receipts.json#L105) 插入斜距 56／100、回环 257／249；[实采 R](../sonar_navigation_integration_audit/evidence/paired_run_receipts.json#L179) 两类因子都是 0／0。插边记录没有逐行区间分支和条件数，不能推断所有行激活，也不能把数量与导航增益直接等同。主仿真的 RMSE 以历史 JSON 的定义为准，未与 HIL 混报。

## 附：独立雅可比探针复跑

在仓库根目录执行。使用 F 的模块，不修改源码和数据；若未归档 F，请先核对本页列出的三个 SHA-256 和第二阶段运行时清单，再复跑。为避免 GTSAM 绑定返回的矩阵视图在临时线性因子释放后失效，显式保留线性因子对象，并复制矩阵。

~~~bash
audit_rt='output/论文数据资料库_20261124/02_匹配导航算法/复现实验_冻结导航运行时/runtime'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$audit_rt/standalone:$audit_rt/core" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-sonar-denoise/bin/python - <<'PY'
import numpy as np, gtsam
from types import SimpleNamespace
from SideScanMappedRangeChange import MappedRangeChange as M, profiled_range_residual_jacobians as fun, world_origins
from SideScanMappedRangeGraph import MappedRangePoseGraph
rng=np.random.default_rng(731); h=1e-6; G=MappedRangePoseGraph()
worst=np.zeros(3); scaled=np.zeros(3); max_cost=0.; max_chart=0.; skipped=active=0
def vals(pa,pb):
    V=gtsam.Values(); V.insert(G._key(0),pa); V.insert(G._key(1),pb); return V
for i in range(120):
    n=17; a=np.r_[rng.normal(0,3,2),rng.uniform(-2,2)]; b=np.r_[rng.normal(0,3,2)+[8,30],rng.uniform(-2,2)]
    L=rng.uniform([-60,20],[60,110],(n,2)); u=rng.uniform(-5,5,(n,2)); w=rng.uniform(-5,5,(n,2)); Z=-rng.uniform(18,80,n)
    za=rng.normal(0,.5,n); dz=[0,1e-6,-1e-6,.25,-.25,12.,-12.][i%7]; zb=za+dz
    ra=np.sqrt(np.sum((L-world_origins(a,u)[0])**2,1)+(Z-za)**2); rb=np.sqrt(np.sum((L-world_origins(b,w)[0])**2,1)+(Z-zb)**2)
    if i%3: a+=rng.normal(0,[4,6,.1]); b+=rng.normal(0,[4,6,.1])
    m=M(L,u,w,za,zb,ra,rb,rng.uniform(.7,4,n))
    va=L-world_origins(a,u)[0]; vb=L-world_origins(b,w)[0]
    D=np.sum(vb**2-va**2,1)-rb**2+ra**2; mid=(za+zb)/2
    L0=2*(zb-za)*(np.minimum(za,zb)-m.maximum_range_m-3*m.pressure_std_m-mid); H0=2*(zb-za)*(np.minimum(za,zb)+3*m.pressure_std_m-mid)
    lo,hi=np.minimum(L0,H0),np.maximum(L0,H0)
    ok=(lo==hi)|(np.minimum(abs(D-lo),abs(D-hi))>1e-3)
    skipped+=int((~ok).sum()); active+=int((ok&((D<lo)|(D>hi)|(hi==lo))).sum())
    for k in range(3):
        if k<2:
            val,ja,jb=fun(a,b,m,robust=bool(k)); A=np.c_[ja,jb]
        else:
            pa,pb=gtsam.Pose2(*a),gtsam.Pose2(*b); edge=SimpleNamespace(kind='mapped_range_change',source=0,target=1)
            G.mapped_range_changes[id(edge)]=m; F=G._edge_factor(edge); V=vals(pa,pb)
            J=F.linearize(V); A=np.array(J.getA(),copy=True); E=F.unwhitenedError(V)
            max_cost=max(max_cost,abs(F.error(V)-E@E/2))
            _,ja,jb=fun(a,b,m)
            max_chart=max(max_chart,np.max(abs(A-np.c_[ja@G._local_delta_to_world(a),jb@G._local_delta_to_world(b)])))
        for j in range(6):
            d=np.eye(3)[j%3]*h
            if k<2:
                ep=fun(a+d,b,m,robust=bool(k))[0] if j<3 else fun(a,b+d,m,robust=bool(k))[0]
                en=fun(a-d,b,m,robust=bool(k))[0] if j<3 else fun(a,b-d,m,robust=bool(k))[0]
            else:
                ep=F.unwhitenedError(vals(pa.retract(d),pb) if j<3 else vals(pa,pb.retract(d)))
                en=F.unwhitenedError(vals(pa.retract(-d),pb) if j<3 else vals(pa,pb.retract(-d)))
            err=abs((ep-en)/(2*h)-A[:,j])[ok]
            worst[k]=max(worst[k],np.max(err,initial=0.)); scaled[k]=max(scaled[k],np.max(err/(1+abs(A[ok,j])),initial=0.))
print('TOTAL', 120*17,'SKIP',skipped,'ACTIVE',active,'ABS',worst,'SCALED',scaled,'COST',max_cost,'CHART',max_chart)
assert max(worst) < 2e-7 and max_cost<1e-11 and max_chart<1e-12
PY
~~~
