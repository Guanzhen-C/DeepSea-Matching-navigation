# 网页可审阅的证据

- [配对运行回执](paired_run_receipts.json)：仅包含原文件摘要、冻结运行时文件计数、各臂诊断和配对后的统计。重算 R 的 DR 时只取奇索引查询；仿真取全程。相同 ping 上对 `map_epoch` 配对，接受数量与匹配/内点中位数分开。
- [原行号源码摘录](SOURCE_EXCERPTS.md)：记录实采的接收入口、仿真包构造、导航输入、子图几何和图因子。第一阶段另有[模型执行与实跑壳层摘录](../../sonar_denoising_model_audit/evidence/SOURCE_EXCERPTS.md)。
- [上一阶段的定位汇总](../../sonar_denoising_model_audit/evidence/navigation_aggregates.csv)：三批各臂的 DR/Local/Global 误差统计，未混合仿真真值和同期 GNSS 参考。

回执不含原始样点、每次输出的坐标、真值轨迹、场景栅格或私人网络信息。哈希用于关联原归档，不表示原档已公开；源码摘录也不是可独立部署的完整目录。档案索引、核查边界和后续安排见[总览](../README.md)。
