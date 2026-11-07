# 海底地形地貌匹配算法仿真报告阅读说明

## 一、先读哪些文件

1. [报告正文](海底地形地貌匹配算法仿真报告.md)：试验条件、算法、指标、结果和局限。
2. [完整指标JSON](data/verified_metrics.json)：重新从全部轨迹计算的误差指标。
3. [数据字典](数据字典.md)：各数据列、时间轴、参考坐标和压缩形式。
4. [来源清单](evidence/export_manifest.json)：全部导出文件的字节数、来源和SHA256。
5. [main算法版本索引](../../../config/validated_algorithm_profiles.json)：确定报告讨论的具体版本。

[Word可编辑版](海底地形地貌匹配算法仿真报告.docx)与[PDF阅读版](海底地形地貌匹配算法仿真报告.pdf)由仓库Markdown源稿生成。所有正文图表使用本目录`figures/`，不会引用只在本机存在的图片链接。

## 二、网页重写报告的建议阅读顺序

先读取正文、数据字典和`verified_metrics.json`，再读取三个`*_segments_100m.csv`查看航程变化，最后按需要读取特定证据JSON和算法源文件。完整CSV较长，不应把网页预览的前几行当成全部数据，也不要在未读完整数据时自行重算全程分位值。

可采用以下任务说明：

```text
请仅以当前main分支为依据，重写《海底地形地貌匹配算法仿真报告》。
先读本目录报告正文、数据字典、verified_metrics.json、export_manifest.json和
config/validated_algorithm_profiles.json，再检查HIL与纯侧扫实际证据和对应算法。
分别阐述海湖处理器在环、实测声呐去噪、纯侧扫导航三部分，不混合不同观测条件。
保留海区跨架构退化、持续实时性、去噪无真实clean标签、纯侧扫弱约束、
准确初值、同航次名义地理编码及尚未核验100%像素覆盖等限制。
不要把71.44%的DR对比增益说成去噪的独立贡献，不把单元测试说成实船验收。
正文使用题目为一级标题，二级“一、”，三级“1.”，四级“（1）”。
所有指标引用具体证据文件，新增计算明确给出公式、样本范围和结果来源。
```

## 三、完整数值与网页易读数据

| 数据 | 样本数 | 完整误差 | 完整轨迹 | 易读统计 |
|---|---:|---|---|---|
| 湖区板载 | 157056 | `lake_hil_errors_full.csv` | `lake_hil_trajectories_full.csv.gz` | `lake_hil_segments_100m.csv` |
| 海区板载 | 243569 | `sea_hil_errors_full.csv` | `sea_hil_trajectories_full.csv.gz` | `sea_hil_segments_100m.csv` |
| 千岛湖纯侧扫 | 62458 | `qiandao_side_scan_errors_full.csv` | `qiandao_side_scan_trajectories_full.csv.gz` | `qiandao_side_scan_segments_100m.csv` |

上述文件位于`data/`。总计463083个同步误差样本，未按终端打印间隔抽样。gzip只压缩字节，不降低精度、不删行；解压后是标准UTF-8 CSV。表中的两个HIL文件也保留同时间戳的原PC误差，用于跨架构比较。

## 四、哪些资源没有上传

原始MAT、数十GB海区原始传感器归档、完整原生I/Q序列、超限原尺寸PNG、各个人运行环境及本机私有配置没有塞入本目录。它们的来源、范围或哈希见证据记录。该限制不影响重算这里的误差和轨迹，但从头生成传感器仿真需要另行取得相应场景和观测资源。

## 五、构建与检查

数据导出程序为`standalone/BuildSeabedSimulationReportEvidence.py`，从本机原归档读取并重新核对全部样本；输出目录已有数据时拒绝覆盖。Word/PDF构建程序为`standalone/BuildSeabedSimulationReportDocument.py`。

报告构建Python依赖见[`requirements-report.txt`](../../../requirements-report.txt)，本次实际版本见[`document_environment.json`](evidence/document_environment.json)。PDF转换与检查另外使用LibreOffice、Poppler。转换进程隔离仿真环境的库路径，不修改原导航环境。重新构建时请使用新的输出副本，不覆盖已经审查的版本。

交付审查程序为`standalone/AuditSeabedSimulationReport.py`，会重新读取全部误差CSV和完整压缩XYZ轨迹、核对51份导出产物哈希、检查本地链接及Word/PDF内容。实际审查结果见[`delivery_audit.json`](evidence/delivery_audit.json)。

报告使用本次实际数据；不得以以前研究报告中摘录的旧统计替换本目录JSON。算法、数据及正文后续若有修改，应重新核验来源哈希、完整指标和文档，再提交main。
