# Wallops and Assawoman Islands USGS 2024 acoustic map

本目录保存 USGS 数据发布 [10.5066/P14SGUMT](https://doi.org/10.5066/P14SGUMT) 的完整多波束测深与后向散射产品。调查于 2024 年 6 月 1 日至 13 日实施，使用两台 Teledyne Reson SeaBat T50-P 组成双头多波束系统。

## 已下载内容

- `source_archives/Wallops_2024_MBES_xyz.zip`：两个坐标/垂直基准版本的 1 m CUBE 测深节点。
- `source_archives/Wallops_2024_MBES_Backscatter.zip`：多波束声学后向散射 GeoTIFF。
- `raw/2024-310-FA_MBES_Wallops_1m_WGS84_UTM18N.xyz`：WGS84 (G2139) / UTM 18N，椭球高为垂向值。
- `raw/2024-310-FA_MBES_Wallops_1m_NAD83_NAVD88_VAStatePlane_USSFT.txt`：NAD83 Virginia South State Plane，NAVD88/GEOID18，高程单位为美国测量英尺。
- `raw/2024-310-FA_Wallops_MBES_1m_Backscatter.tiff`：float32、LZW 压缩的后向散射镶嵌图。
- `raw/Wallops_2024_MBES_metadata.{txt,xml}` 与官方预览图。

两份测深文本是同一批 34,649,380 个节点的不同坐标表达，不应当作两次独立观测重复融合。本地扫描 WGS84 文件得到：

- UTM 范围：E 453332.5–464689.5 m，N 4178775.5–4194493.5 m；
- 椭球高：-51.96 至 -39.60 m；
- CUBE 不确定度：均值 0.049106 m，最大值 0.575153 m；
- 官方元数据给出的平均 CUBE 不确定度约 0.05 m，VDatum 转换另增加约 0.0370 m 垂向误差。

## 科学注意事项

- 官方产品描述为 1 m CUBE 网格；实际后向散射 TIFF 标签显示 0.2 m 像元间距、79882 × 59402 的稀疏画布。该 0.2 m 发布像元不能视为独立声学分辨率。与测深融合前建议保留 NoData 掩膜并聚合/重采样到共同网格。
- 官方后向散射强度范围为 -65.625 至 9.6875 dB；它是 SeaBat T50-P 的多波束后向散射，不是独立侧扫声呐。
- 有效测深的垂向起伏约 12.36 m，适合平坦/低起伏区域的退化与鲁棒性测试，不足以单独支撑 73 km 不重复航程。
- 原始数据按 CC0 1.0 发布、属于公有领域；不得用于实际航行。
- 发布页没有提供压缩包哈希；`source_sha256sums.txt` 是本次下载后本地计算的 SHA-256，且两个 ZIP 均已通过 `unzip -t`。

官方发布页：<https://coastal.er.usgs.gov/data-release/doi-P14SGUMT/>
