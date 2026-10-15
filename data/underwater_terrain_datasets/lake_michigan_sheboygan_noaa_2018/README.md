# Lake Michigan Sheboygan NOAA 2018 acoustic map

本目录保存 NOAA/NCEI accession **0222391** 中的 Sheboygan 2018 测区（测量编号 W00463）。这是当前集合里唯一同时包含独立多波束测深、多波束后向散射和专业双频侧扫声呐镶嵌图的数据包。

## 传感器与采集

- Kongsberg EM2040C：多波束测深及 snippets 后向散射；
- Klein System 3000 拖鱼侧扫：100 kHz 与 500 kHz 同时采集，每侧量程 75 m，原始横向分辨率约 0.25 m；
- 调查船为 NOAA R/V Storm；W00463 报告所列正式测量日期为 2018-06-02 至 2018-06-11；
- 多波束与侧扫沿相同调查作业同时采集，但侧扫拖鱼位于船后约 50–150 m，不能当作与多波束刚性同位的传感器。

## 主要文件

解压数据位于 `raw/Sheboygan_2018/`：

- 浅水/深水多波束测深：2 m / 4 m GeoTIFF；
- 浅水/深水多波束后向散射：1 m / 2 m GeoTIFF；
- 高频/低频侧扫镶嵌图：0.5 m GeoTIFF；
- 测深派生的坡度、坡度变化率和标准差栅格；
- 对应的世界文件、金字塔和 ArcGIS/GDAL 元数据。

官方总数据说明、NCEI ISO 元数据和 W00463 调查报告分别保存在 `raw/DataDocumentation.pdf`、`raw/EXGF29-ISO-19115-2.xml` 和 `raw/W00463_DR.pdf`。

## 科学注意事项

- 多波束水平位置经 patch test 估计优于约 0.57 m；官方说明明确指出侧扫拖鱼位置不确定度未知且明显更大，最终侧扫镶嵌图利用多波束可见底部特征进行了细化配准。
- 文档称侧扫输出为 WGS84 / UTM 16N，但本包高频侧扫 TIFF 实际自带 NAD83 / UTM 18N（EPSG:26918）坐标，负 Easting 是因为测区远在该分区中央经线以西；低频侧扫为 NAD83 / UTM 16N（EPSG:26916）。应执行真实坐标变换，不能直接覆盖 CRS 标签。
- 测深主图为 NAD83 / UTM 16N；多波束后向散射主图为 WGS84 / UTM 16N。全部数据需要统一 CRS、垂向基准和像元网格后才能联合使用。
- 浅水后向散射 TIFF 的 NoData 标签写为 256，但其存储类型为 uint8，256 无法实际表示。预处理时应依据覆盖范围/有效掩膜识别背景，不能只比较该 NoData 数值。
- 文件名包含 `EM2040c_Sidescan`，但官方数据说明和调查报告均确认独立侧扫由 Klein System 3000 获取；应以元数据中的传感器记录为准。
- 这是处理后的 GeoTIFF 地图包，不包含逐 ping 的 `.all`、`.xtf` 或 `.sdf` 原始时序。
- `source_sha256sums.txt` 是本次下载后本地计算的 SHA-256；官方 ZIP 已通过 `unzip -t`。

官方 accession：<https://www.ncei.noaa.gov/archive/archive-management-system/OAS/bin/prd/jquery/accession/details/222391>
