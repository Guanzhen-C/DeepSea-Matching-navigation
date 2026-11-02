# 研究报告 LaTeX 修订工作区（2026-10-03）

本目录在 2026-10-02 的四份《纯文献深化版》报告基础上修订；原报告及原压缩包不改动。四份报告现已统一使用同一套 LaTeX 模板。收到并全文核读的第 1 篇论文直接涉及 3 月导航调研与 6 月全局重定位，故本轮只对这两份作证据性内容更新；5 月超分、7 月声呐里程计报告先保留基线论证并完成 LaTeX 重排，等待相应文献后继续补充。

## 本次证据

- Zhang Q, Kim J. *Terrain-Based Global Tracking Without Prior Pose Information Using Multibeam Echosounder Measurements*. IEEE Transactions on Instrumentation and Measurement, 2026, 75:7506113. DOI:10.1109/TIM.2026.3716492。
- 输入为用户提供的 13 页正式论文 PDF；SHA-256：`0364546bf0c87979e037f2146d4a1e6d714298511d6b3112ac710fb14a3758f5`。原论文不复制进本交付目录。
- 6 月报告新增 10.3 节，区分原文算法、原文实验数字和本报告的研究建议；3 月报告修正 4.4 节及相应参考文献、阅读范围。

## 文件及重建

- `03月*.md`、`05月*.md`、`06月*.md`、`07月*.md`：原报告或本次修订稿的编辑参考；`assets/` 为原报告插图。
- `report_03.tex`、`report_05.tex`、`report_06.tex`、`report_07.tex`：四份 LaTeX 入口；`research_report.sty` 为共同模板。
- 对应 `report_XX_body.tex`：当前**排版和后续内容修订的主源文件**。它们由对应 Markdown 转换，再对宽表、长 DOI 加入断行处理；重新从 Markdown 生成时，需重新核查这些排版修正，不能直接覆盖已有 TeX 源文件。
- 对应 `report_XX.pdf`：四份 PDF。可在本目录分别运行 `tectonic report_XX.tex`（或使用兼容的 XeLaTeX 工具链）重建。

5 月和 7 月的 TeX 正文生成时跳过了原 Markdown 的标题及手工目录，改由 LaTeX 模板统一生成标题和目录；正文论证、图表、参考文献和阅读范围保持原有内容，不把重排称作新文献研究成果。

## 后续补充约定

每篇新论文先核对全文和原表/图，再记录来源与阅读范围；陈述时分清原文方法、实验条件/指标与本报告推论。只读到摘要的文献不补写未见的公式、参数或实验结果。跨论文比较先统一场景、数据、初值和指标定义，不拼凑排行榜。新证据回填到相关的既有章节、参考文献与阅读范围表，并重新编译、检查 PDF，不另以独立读书笔记充当研究报告。
