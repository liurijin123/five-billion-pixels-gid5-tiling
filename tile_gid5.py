"""从 Five-Billion-Pixels 生成 256×256 的 GID5 语义分割样本。

程序保留 GID5 原始标签编码，不进行重映射：

    0 = 未标注／ignore_index
    1 = 建成区，2 = 农田，3 = 森林，4 = 草地，5 = 水体

使用本程序输出训练模型时，模型头应输出六个通道，并配合
``torch.nn.CrossEntropyLoss(ignore_index=0)`` 使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window


PROGRAM_VERSION = "1.0.0"
TILE_SIZE = 256
STRIDE = 256
EXPECTED_SCENE_COUNT = 150
CLASS_NAMES = {
    0: "unlabeled_ignore",
    1: "built_up",
    2: "farmland",
    3: "forest",
    4: "meadow",
    5: "water",
}
MANIFEST_FIELDS = [
    "image_path",
    "label_path",
    "scene_id",
    "row_off",
    "col_off",
    "valid_fraction",
    "pixels_class_0",
    "pixels_class_1",
    "pixels_class_2",
    "pixels_class_3",
    "pixels_class_4",
    "pixels_class_5",
]
LABEL_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 255],
        [255, 255, 0],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Scene:
    """表示一对已经通过检查的本地 GID5 影像与标签。"""

    scene_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int

    @property
    def candidate_tiles(self) -> int:
        return (self.width // TILE_SIZE) * (self.height // TILE_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Rasterio 生成无重叠的 256×256 GID5 训练样本。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="包含 Image__8bit_NirRGB/ 和 GID5/Annotation__index/ 的数据集根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="用于保存切块和清单的新目录或可断点续跑目录。",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise RuntimeError(message)


def atomic_replace_path(path: Path, writer: Callable[[Path], None]) -> None:
    """先在目标文件旁写入临时文件，再以原子操作发布为正式文件。

    程序不会覆盖已经存在的正式文件。若发现遗留的 ``.part`` 文件，说明
    之前的运行可能中断；继续处理前必须由用户检查，程序不会自行删除。
    这样可以保留诊断依据，避免把不完整文件误认为正式结果。
    """

    if path.exists():
        fail(f"拒绝覆盖已有文件：{path}")
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        fail(f"检测到未完成的临时文件，请先检查后处理：{temporary}")

    try:
        writer(temporary)
        os.replace(temporary, path)
    except Exception:
        # 保留 .part 文件用于诊断，程序绝不静默删除它。
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if path.exists():
        temporary = path.with_name(path.name + ".part")
        if temporary.exists():
            fail(f"检测到未完成的临时文件，请先检查后处理：{temporary}")
        write(temporary)
        os.replace(temporary, path)
    else:
        atomic_replace_path(path, write)


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)

    def write(temporary: Path) -> None:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    if path.exists():
        temporary = path.with_name(path.name + ".part")
        if temporary.exists():
            fail(f"检测到未完成的临时文件，请先检查后处理：{temporary}")
        write(temporary)
        os.replace(temporary, path)
    else:
        atomic_replace_path(path, write)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != MANIFEST_FIELDS:
            fail(f"分景清单字段不符合本程序版本：{path}")
        return list(reader)


def open_raw_image(path: Path) -> rasterio.io.DatasetReader:
    """读取 TIFF 中的四个存储通道，避免 GDAL 执行 CMYK 到 RGBA 的转换。

    发布的 8 位 TIFF 被标记为 CMYK，但数据集目录说明四个存储通道依次为
    NIR、R、G、B。使用常规 GDAL/Rasterio 方式打开时，这些通道会被转换为
    RGBA，第四波段也会变成不透明的 Alpha 通道。添加 GTIFF_RAW 前缀可保留
    文件中实际存储的字节。
    """

    return rasterio.open("GTIFF_RAW:" + str(path.resolve()))


def ensure_output_layout(dataset_root: Path, output_root: Path) -> dict[str, Path]:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()

    if dataset_root == output_root or dataset_root in output_root.parents:
        fail("输出目录不得是数据集目录本身或其子目录，以免混入原始数据。")

    output_root.mkdir(parents=True, exist_ok=True)
    layout = {
        "root": output_root,
        "images": output_root / "images",
        "labels": output_root / "labels",
        "manifests": output_root / "manifests",
        "preview": output_root / "preview",
    }
    for directory in layout.values():
        if directory != output_root:
            directory.mkdir(parents=True, exist_ok=True)
    return layout


def collect_and_preflight(dataset_root: Path) -> list[Scene]:
    image_dir = dataset_root / "Image__8bit_NirRGB"
    label_dir = dataset_root / "GID5" / "Annotation__index"
    if not image_dir.is_dir() or not label_dir.is_dir():
        fail(
            "数据集目录结构不符合预期。需要 Image__8bit_NirRGB/ 和 "
            "GID5/Annotation__index/。"
        )

    images = sorted(image_dir.glob("*.tif"))
    labels = sorted(label_dir.glob("*_5label.png"))
    if len(images) != EXPECTED_SCENE_COUNT or len(labels) != EXPECTED_SCENE_COUNT:
        fail(
            f"预期各有 {EXPECTED_SCENE_COUNT} 个影像和标签，实际为 "
            f"{len(images)} 个影像、{len(labels)} 个标签。"
        )

    image_by_scene = {path.stem: path for path in images}
    label_by_scene = {path.stem.removesuffix("_5label"): path for path in labels}
    if set(image_by_scene) != set(label_by_scene):
        missing_labels = sorted(set(image_by_scene) - set(label_by_scene))
        extra_labels = sorted(set(label_by_scene) - set(image_by_scene))
        fail(
            "影像和标签未一一配对。"
            f" 缺失标签：{missing_labels[:5]}；多余标签：{extra_labels[:5]}。"
        )

    scenes: list[Scene] = []
    print("[1/3] 正在预检 150 对影像和标签……", flush=True)
    for ordinal, scene_id in enumerate(sorted(image_by_scene), start=1):
        image_path = image_by_scene[scene_id]
        label_path = label_by_scene[scene_id]
        with open_raw_image(image_path) as image_src, rasterio.open(label_path) as label_src:
            if image_src.count != 4:
                fail(f"影像不是预期的 4 波段：{image_path}（实际 {image_src.count} 波段）")
            if label_src.count != 1:
                fail(f"标签不是预期的单波段：{label_path}（实际 {label_src.count} 波段）")
            if (image_src.width, image_src.height) != (label_src.width, label_src.height):
                fail(
                    f"影像和标签尺寸不一致：{image_path.name} 为 "
                    f"{image_src.width}×{image_src.height}，{label_path.name} 为 "
                    f"{label_src.width}×{label_src.height}。"
                )
            if image_src.dtypes != ("uint8",) * 4:
                fail(f"影像数据类型不是 8 位无符号整数：{image_path}")
            if label_src.dtypes != ("uint8",):
                fail(f"标签数据类型不是 8 位无符号整数：{label_path}")

            seen_values = set(np.unique(label_src.read(1)).tolist())
            unexpected = sorted(seen_values - set(CLASS_NAMES))
            if unexpected:
                fail(f"标签存在不在 0–5 中的类别值：{label_path}，值为 {unexpected}")

            scenes.append(
                Scene(
                    scene_id=scene_id,
                    image_path=image_path,
                    label_path=label_path,
                    width=image_src.width,
                    height=image_src.height,
                )
            )
        print(f"  已预检 {ordinal:>3}/{EXPECTED_SCENE_COUNT}: {scene_id}", flush=True)
    return scenes


def output_paths(layout: dict[str, Path], scene_id: str, row_off: int, col_off: int) -> tuple[Path, Path]:
    tile_name = f"{scene_id}_r{row_off:04d}_c{col_off:04d}"
    image_path = layout["images"] / scene_id / f"{tile_name}.tif"
    label_path = layout["labels"] / scene_id / f"{tile_name}.png"
    return image_path, label_path


def validate_existing_pair(image_path: Path, label_path: Path) -> None:
    with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
        if image_src.count != 4 or (image_src.width, image_src.height) != (TILE_SIZE, TILE_SIZE):
            fail(f"已有影像块结构不正确：{image_path}")
        if label_src.count != 1 or (label_src.width, label_src.height) != (TILE_SIZE, TILE_SIZE):
            fail(f"已有标签块结构不正确：{label_path}")
        values = set(np.unique(label_src.read(1)).tolist())
        if values - set(CLASS_NAMES):
            fail(f"已有标签块含非法类别值：{label_path}")


def write_image_tile(path: Path, image_src: rasterio.io.DatasetReader, window: Window) -> None:
    image_tile = image_src.read(window=window)
    if image_tile.shape != (4, TILE_SIZE, TILE_SIZE):
        fail(f"影像窗口尺寸异常：{image_src.name}，窗口 {window}")

    profile = image_src.profile.copy()
    profile.update(
        driver="GTiff",
        width=TILE_SIZE,
        height=TILE_SIZE,
        count=4,
        dtype="uint8",
        transform=image_src.window_transform(window),
        tiled=True,
        blockxsize=TILE_SIZE,
        blockysize=TILE_SIZE,
        compress="deflate",
        predictor=2,
        interleave="band",
        photometric="MINISBLACK",
        BIGTIFF="IF_SAFER",
    )

    def write(temporary: Path) -> None:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(image_tile)

    atomic_replace_path(path, write)


def write_label_tile(path: Path, label_tile: np.ndarray) -> None:
    if label_tile.shape != (TILE_SIZE, TILE_SIZE):
        fail(f"标签窗口尺寸异常：{label_tile.shape}")

    profile = {
        "driver": "PNG",
        "width": TILE_SIZE,
        "height": TILE_SIZE,
        "count": 1,
        "dtype": "uint8",
    }

    def write(temporary: Path) -> None:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(label_tile, 1)

    atomic_replace_path(path, write)


def row_from_tile(
    layout: dict[str, Path],
    image_path: Path,
    label_path: Path,
    scene_id: str,
    row_off: int,
    col_off: int,
    label_tile: np.ndarray,
) -> dict[str, Any]:
    counts = np.bincount(label_tile.ravel(), minlength=len(CLASS_NAMES))
    return {
        "image_path": image_path.relative_to(layout["root"]).as_posix(),
        "label_path": label_path.relative_to(layout["root"]).as_posix(),
        "scene_id": scene_id,
        "row_off": row_off,
        "col_off": col_off,
        "valid_fraction": f"{1.0 - counts[0] / label_tile.size:.8f}",
        **{f"pixels_class_{class_id}": int(counts[class_id]) for class_id in CLASS_NAMES},
    }


def process_scene(scene: Scene, layout: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_manifest = layout["manifests"] / f"{scene.scene_id}.csv"
    image_directory = layout["images"] / scene.scene_id
    label_directory = layout["labels"] / scene.scene_id
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)

    if scene_manifest.exists():
        rows = read_csv_rows(scene_manifest)
        for row in rows:
            validate_existing_pair(
                layout["root"] / row["image_path"],
                layout["root"] / row["label_path"],
            )
        return rows, {
            "scene_id": scene.scene_id,
            "candidate_tiles": scene.candidate_tiles,
            "written_tiles": len(rows),
            "skipped_empty_tiles": scene.candidate_tiles - len(rows),
            "resumed_completed_scene": True,
        }

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    with open_raw_image(scene.image_path) as image_src, rasterio.open(scene.label_path) as label_src:
        label_array = label_src.read(1)
        for row_off in range(0, scene.height - TILE_SIZE + 1, STRIDE):
            for col_off in range(0, scene.width - TILE_SIZE + 1, STRIDE):
                window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
                label_tile = label_array[
                    row_off : row_off + TILE_SIZE,
                    col_off : col_off + TILE_SIZE,
                ]
                if label_tile.shape != (TILE_SIZE, TILE_SIZE):
                    fail(f"标签窗口尺寸异常：{scene.label_path}，窗口 {window}")
                if not np.any(label_tile):
                    skipped_empty += 1
                    continue
                if set(np.unique(label_tile).tolist()) - set(CLASS_NAMES):
                    fail(f"标签块含非法类别值：{scene.label_path}，窗口 {window}")

                image_path, label_path = output_paths(layout, scene.scene_id, row_off, col_off)
                image_exists = image_path.exists()
                label_exists = label_path.exists()
                if image_exists != label_exists:
                    fail(
                        "发现不完整的既有切块对，拒绝继续写入："
                        f"影像={image_path}，标签={label_path}。"
                    )
                if image_exists:
                    validate_existing_pair(image_path, label_path)
                else:
                    write_image_tile(image_path, image_src, window)
                    write_label_tile(label_path, label_tile)
                    validate_existing_pair(image_path, label_path)

                rows.append(
                    row_from_tile(
                        layout,
                        image_path,
                        label_path,
                        scene.scene_id,
                        row_off,
                        col_off,
                        label_tile,
                    )
                )

    write_csv_atomic(scene_manifest, rows)
    return rows, {
        "scene_id": scene.scene_id,
        "candidate_tiles": scene.candidate_tiles,
        "written_tiles": len(rows),
        "skipped_empty_tiles": skipped_empty,
        "resumed_completed_scene": False,
    }


def write_png(path: Path, data: np.ndarray) -> None:
    if data.ndim != 3 or data.shape[1:] != (TILE_SIZE, TILE_SIZE):
        fail(f"预览影像形状错误：{data.shape}")
    profile = {
        "driver": "PNG",
        "width": TILE_SIZE,
        "height": TILE_SIZE,
        "count": data.shape[0],
        "dtype": "uint8",
    }

    if path.exists():
        with rasterio.open(path) as existing:
            if existing.count == data.shape[0] and (existing.width, existing.height) == (TILE_SIZE, TILE_SIZE):
                return
        fail(f"已有预览文件结构不正确，拒绝覆盖：{path}")

    def write(temporary: Path) -> None:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(data)

    atomic_replace_path(path, write)


def preview_score(row: dict[str, Any]) -> tuple[int, int, int]:
    counts = [int(row[f"pixels_class_{class_id}"]) for class_id in range(1, 6)]
    diversity = sum(count > 0 for count in counts)
    valid_pixels = sum(counts)
    # 优先选择包含多种类别的有效样本，其次选择有效标注像元更多的样本。
    return diversity, valid_pixels, -int(row["row_off"]) - int(row["col_off"])


def write_article_preview(layout: dict[str, Path], rows: list[dict[str, Any]]) -> dict[str, str] | None:
    if not rows:
        return None
    selected = max(rows, key=preview_score)
    image_path = layout["root"] / selected["image_path"]
    label_path = layout["root"] / selected["label_path"]
    preview_stem = Path(selected["image_path"]).stem

    with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
        rgb = image_src.read([2, 3, 4])
        labels = label_src.read(1)
    label_rgb = np.moveaxis(LABEL_COLORS[labels], -1, 0)
    overlay = rgb.copy()
    labeled = labels != 0
    overlay[:, labeled] = (
        rgb[:, labeled].astype(np.uint16) * 55 + label_rgb[:, labeled].astype(np.uint16) * 45
    ) // 100
    overlay = overlay.astype(np.uint8)

    rgb_path = layout["preview"] / f"{preview_stem}_rgb.png"
    label_preview_path = layout["preview"] / f"{preview_stem}_label.png"
    overlay_path = layout["preview"] / f"{preview_stem}_overlay.png"
    write_png(rgb_path, rgb)
    write_png(label_preview_path, label_rgb)
    write_png(overlay_path, overlay)
    return {
        "source_tile": selected["image_path"],
        "rgb": rgb_path.relative_to(layout["root"]).as_posix(),
        "label": label_preview_path.relative_to(layout["root"]).as_posix(),
        "overlay": overlay_path.relative_to(layout["root"]).as_posix(),
    }


def check_configuration(layout: dict[str, Path], dataset_root: Path) -> None:
    config_path = layout["root"] / "run_config.json"
    expected = {
        "program_version": PROGRAM_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "image_directory": "Image__8bit_NirRGB",
        "label_directory": "GID5/Annotation__index",
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "edge_policy": "drop_incomplete_right_and_bottom",
        "empty_label_policy": "skip_tiles_with_only_label_0",
        "label_encoding": CLASS_NAMES,
        "training_contract": "six_output_channels_with_CrossEntropyLoss(ignore_index=0)",
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != expected:
            fail(f"输出目录的运行配置与当前任务不一致：{config_path}")
        return
    write_json_atomic(config_path, expected)


def write_global_manifest(layout: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    output_path = layout["root"] / "manifest.csv"
    write_csv_atomic(output_path, rows)


def main() -> int:
    arguments = parse_args()
    dataset_root = arguments.dataset_root.expanduser()
    output_root = arguments.output_root.expanduser()
    if not dataset_root.is_dir():
        fail(f"数据集根目录不存在：{dataset_root}")

    started = time.perf_counter()
    scenes = collect_and_preflight(dataset_root)
    layout = ensure_output_layout(dataset_root, output_root)
    check_configuration(layout, dataset_root)

    print("[2/3] 开始全量切块。中断后可用同一命令继续运行。", flush=True)
    all_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for ordinal, scene in enumerate(scenes, start=1):
        scene_started = time.perf_counter()
        rows, scene_summary = process_scene(scene, layout)
        scene_summary["elapsed_seconds"] = round(time.perf_counter() - scene_started, 2)
        scene_summaries.append(scene_summary)
        all_rows.extend(rows)
        action = "已核验并跳过" if scene_summary["resumed_completed_scene"] else "完成"
        print(
            f"  [{ordinal:>3}/{len(scenes)}] {action} {scene.scene_id}："
            f"输出 {scene_summary['written_tiles']}，跳过空块 {scene_summary['skipped_empty_tiles']}。",
            flush=True,
        )

    all_rows.sort(key=lambda row: (row["scene_id"], int(row["row_off"]), int(row["col_off"])))
    print("[3/3] 正在写入总清单、汇总与文章预览……", flush=True)
    write_global_manifest(layout, all_rows)
    preview = write_article_preview(layout, all_rows)

    image_count = sum(1 for _ in layout["images"].rglob("*.tif"))
    label_count = sum(1 for _ in layout["labels"].rglob("*.png"))
    if image_count != len(all_rows) or label_count != len(all_rows):
        fail(
            "输出文件数量与总清单不一致："
            f"影像 {image_count}，标签 {label_count}，清单 {len(all_rows)}。"
        )

    totals = Counter()
    for row in all_rows:
        for class_id in CLASS_NAMES:
            totals[class_id] += int(row[f"pixels_class_{class_id}"])
    summary = {
        "program_version": PROGRAM_VERSION,
        "completed_at": utc_now(),
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(layout["root"]),
        "scene_count": len(scenes),
        "candidate_tile_count": sum(scene.candidate_tiles for scene in scenes),
        "written_tile_count": len(all_rows),
        "skipped_empty_tile_count": sum(item["skipped_empty_tiles"] for item in scene_summaries),
        "image_file_count": image_count,
        "label_file_count": label_count,
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "discarded_right_pixels_per_scene": scenes[0].width % TILE_SIZE,
        "discarded_bottom_pixels_per_scene": scenes[0].height % TILE_SIZE,
        "label_encoding": CLASS_NAMES,
        "training_contract": "six_output_channels_with_CrossEntropyLoss(ignore_index=0)",
        "class_pixel_counts_in_written_tiles": {str(key): totals[key] for key in CLASS_NAMES},
        "preview": preview,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scenes": scene_summaries,
    }
    write_json_atomic(layout["root"] / "summary.json", summary)
    print(
        "全量切块完成："
        f"{summary['written_tile_count']} 对样本，"
        f"跳过 {summary['skipped_empty_tile_count']} 个全零标签块。\n"
        f"结果目录：{layout['root']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        # 即使没有空间元数据，GeoTIFF 仍可作为深度学习训练输入。
        import warnings

        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已收到中断信号；已完成分景清单的结果可用同一命令断点续跑。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\n[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1)
