# Five-Billion-Pixels GID5 256×256 切块

配套文章：《遥感语义分割数据准备：Five-Billion-Pixels 的影像与标签切块》

本程序使用 Rasterio 将 Five-Billion-Pixels 的 150 景四波段 8 位影像与 GID5 索引标签同步切成 256×256 的语义分割样本。

## 处理约定

- 影像：`Image__8bit_NirRGB/*.tif`，使用 `GTIFF_RAW:` 保留 TIFF 中四个原始存储波段。
- 标签：`GID5/Annotation__index/*_5label.png`，保持原始 `0～5` 编码。
- 切块：`tile_size=256`、`stride=256`、无重叠。
- 边缘：舍弃右侧和底部不足 256 像素的区域。
- 空块：跳过标签全部为 `0` 的窗口。
- 训练：模型输出 6 个通道，使用 `torch.nn.CrossEntropyLoss(ignore_index=0)`。

标签定义：

| 值 | 类别 |
| --- | --- |
| `0` | 未标注／训练时忽略 |
| `1` | 建成区 |
| `2` | 农田 |
| `3` | 森林 |
| `4` | 草地 |
| `5` | 水体 |

## 适用环境

- Windows 10/11 64 位
- Python 3.13.1
- Rasterio 1.5.1
- GDAL 3.12.4（随 Rasterio wheel 安装）
- NumPy 2.5.2

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
.\.venv\Scripts\python.exe .\tile_gid5.py `
  --dataset-root "<Five-Billion-Pixels目录>" `
  --output-root "<切块结果目录>"
```

本次本地全量验证命令：

```powershell
.\.venv\Scripts\python.exe .\tile_gid5.py `
  --dataset-root "D:\DATA\datasets\Five-Billion-Pixels" `
  --output-root "D:\DATA\datasets\Five-Billion-Pixels-GID5-256"
```

读者应把路径替换为自己的数据目录。程序不会改动原始数据集。

## 输入要求

```text
<dataset-root>/
├─ Image__8bit_NirRGB/<scene_id>.tif
└─ GID5/Annotation__index/<scene_id>_5label.png
```

程序启动时会检查：

- 影像和标签各为 150 景并可按场景编号一一配对；
- 影像为 4 波段 `uint8`，标签为单波段 `uint8`；
- 每对影像和标签尺寸一致；
- 标签值只包含 `0～5`。

任一检查失败时，程序会停止而不是继续生成部分错误结果。

## 输出结构

```text
<output-root>/
├─ images/<scene_id>/*.tif
├─ labels/<scene_id>/*.png
├─ manifests/<scene_id>.csv
├─ preview/
├─ run_config.json
├─ manifest.csv
└─ summary.json
```

- `images/`：四波段 `uint8` GeoTIFF，DEFLATE 无损压缩。
- `labels/`：单波段 `uint8` PNG，标签值保持 `0～5`。
- `manifests/`：每景完成后写入的断点与样本清单。
- `manifest.csv`：全部样本的来源景、原始偏移、有效比例和类别像元数。
- `summary.json`：场景数、候选块数、输出块数、空块数、类别统计和耗时。
- `preview/`：程序自动选择一个混合类别样本，生成 RGB、标签和彩色叠加 PNG。

## 断点续跑

使用相同命令重新运行即可：

- 已有分景清单的场景会逐文件回读核验后跳过；
- 未形成分景清单的场景会检查已有影像—标签对并继续；
- 单边缺失、结构异常、配置变化或残留 `.part` 文件会触发失败；
- 程序不会静默覆盖已有正式切块。

## 成功判据

- `summary.json` 中 `scene_count` 为 `150`；
- `image_file_count`、`label_file_count` 与 `written_tile_count` 相同；
- `manifest.csv` 数据行数等于 `written_tile_count`；
- 输出影像均为 4 波段 256×256，标签均为单波段 256×256；
- 标签只包含 `0～5`。

## 验证状态

- 单块冒烟检查：已通过，四个影像波段与标签回读后逐像元一致。
- 全量验证：已通过，150 景全部完成。
- 候选窗口：109200；写出样本：106947 对；跳过全零标签块：2253。
- 影像文件、标签文件和 `manifest.csv` 数据行均为 106947，分景清单为 150 个，残留 `.part` 文件为 0。
- 总耗时：3086.67 秒（51.44 分钟）；输出占用：17.53 GiB。
- 详细记录：`research/2026-08-12-Five-Billion-Pixels生成256像素GID5分割样本/notes.md`

## 数据与 GitHub

原始数据集和切块结果不属于本代码仓库，也不会上传 GitHub。仓库只发布正式程序、依赖清单和使用说明。

- 仓库：https://github.com/liurijin123/five-billion-pixels-gid5-tiling
- 状态：已发布，`main` 分支远端提交已核对
- 最后验证日期：2026-08-12
