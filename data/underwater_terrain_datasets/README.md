# 水下地形与声呐公开数据集

本目录汇总了四套已经完整下载、解压并完成压缩包完整性检查的公开实测数据。每套数据都已生成 OceanSim 可直接加载的地图、联合有效区掩膜和经过 250 m 声呐边界余量检查的闭合验证航线；它们不会替换默认 Monterey 地图或修改定位算法配置。

## 数据集清单

| 目录 | 实测数据 | 主要分辨率 | 适合用途 | 关键限制 |
|---|---|---:|---|---|
| `lake_chelan_usgs_2023_3m/` | 多波束测深 + 同航次多波束后向散射 | 3 m | 约 80 km 湖区的长航程 SLAM、地形/强度联合匹配 | 强度不是独立侧扫；五个分块需统一配准和归一化 |
| `jenkinson_lake_usgs_2023/` | 干涉式侧扫/测深 + 后向散射 | 1 m，全湖；0.5 m，局部 | 小范围高分辨率单元测试、局部匹配 | 全湖仅约 2.4 km²，不适合 73 km 单程测试 |
| `wallops_assawoman_usgs_2024_1m/` | 双头多波束测深 + 多波束后向散射 | 测深节点 1 m | 高精度绝对地形匹配、低起伏与近岸鲁棒性测试 | 不是独立侧扫；水深起伏约 12.36 m，不能单独承担长航程场景 |
| `lake_michigan_sheboygan_noaa_2018/` | 多波束测深 + 多波束后向散射 + Klein 3000 双频侧扫 | 测深 2/4 m；后向散射 1/2 m；侧扫 0.5 m | 多波束与真实侧扫联合建模、跨模态配准和特征前端验证 | 侧扫为拖鱼系统，位置不确定度高于多波束；各栅格 CRS 不完全一致 |

机器可读清单见 `catalog.json`。每个子目录均包含详细说明和源文件校验清单。

## 目录约定

- `source_archives/`：官方原始压缩包，便于重新解压和追溯。
- `raw/`：未经本项目改写的官方解压文件、元数据和说明文档。
- `prepared/`：统一到 OceanSim 本地坐标系的测深、声学强度、有效区掩膜、元数据及验证航线。
- `source_md5sums.txt`：发布方提供的 MD5（Lake Chelan、Jenkinson Lake）。
- `source_sha256sums.txt`：本次下载后计算的 SHA-256（Wallops、Sheboygan；发布页未提供哈希）。

大体积原始数据由本目录内的 `.gitignore` 排除，避免误提交到 Git；README、目录清单和校验文件仍可纳入版本管理。

## 在成熟算法上运行

首次恢复原始数据后，可统一预处理全部地图：

```bash
cd /home/cgz/OceanSim
python standalone/prepare_public_terrain_datasets.py --dataset all
```

使用经过长航程验证的正式算法分别启动 UI：

```bash
./standalone/run_v2_public_terrain.sh chelan
./standalone/run_v2_public_terrain.sh jenkinson
./standalone/run_v2_public_terrain.sh wallops
./standalone/run_v2_public_terrain.sh sheboygan
```

无界面加速运行时在命令末尾追加 `--headless --time-scale 3600 --terrain-display-stride 32`。运行入口只替换真实地图、配套航线、起点和结果文件；前端、粒子地形跟踪、GTSAM 后端及 map→odom 软校正均沿用正式算法。末尾还可继续追加参数覆盖默认值，例如用 `--survey-segment-distance-m 1000` 做 1 km 短测。

## 使用优先级

1. **73 km 长航程地图：Lake Chelan。** 它是四套数据中唯一在单一湖区尺度上自然覆盖约 80 km、且同时提供测深和声学强度的产品。
2. **真实多波束 + 独立侧扫联合实验：Sheboygan。** 这是传感器组合最接近当前 AUV 目标的一套，但侧扫是拖鱼获取，不是与 AUV 刚性同位安装；仿真时必须显式建模杆臂、拖鱼定位误差或将其仅作为底图纹理。
3. **高精度绝对匹配与平坦区退化实验：Wallops。** 1 m CUBE 测深节点包含逐点不确定度，适合验证粒子滤波/地形匹配置信度设计。
4. **快速、小范围回归测试：Jenkinson Lake。** 数据量较小、分辨率最高，适合迭代预处理与匹配代码。

## 统一预处理要求

- 所有栅格在进入仿真前必须统一水平 CRS、垂直基准、像元网格和 NoData 掩膜。
- 测深栅格分辨率不等同于绝对定位精度；不得直接把像元尺寸当作传感器误差。
- 后向散射和侧扫灰度不是天然统一的绝对反射强度。跨航线、跨频率或跨设备使用前需做增益/角度响应归一化。
- Sheboygan 高频侧扫文件实际携带 NAD83 / UTM 18N（EPSG:26918），其余主要产品多为 UTM 16N；必须按各文件自身 CRS 重投影，不能只改 CRS 标签。
- Wallops 官方称产品来自 1 m 网格，但后向散射 GeoTIFF 的像元标签为 0.2 m；这只是发布栅格的采样间距，不能解释为 0.2 m 独立声学分辨率。与 1 m 测深节点融合时应统一到科学合理的共同网格。
- 这些下载包主要是处理后的地图产品，不是逐 ping 原始声呐时序。它们适合构造真实底图和验证匹配算法，但不能替代声呐传播、波束形成及噪声模型。

## 未纳入项

原有 `data/qiandao_lake/` 是约 30 m 的推断型湖底数据，且没有同源声学强度，因此保留在本集合之外，没有删除。原有 Monterey 数据同样未移动或修改。
