"""Draw random cached VisionKG detection annotations into a PDF.

The VisionKG cache contains images, while run directories contain the COCO
annotation files that refer to them. This utility joins both using each COCO
image's ``image_path`` field and interprets ``bbox`` as COCO ``xywh``.

Run from the backend directory, for example::

    uv run python src/cvmodellearning/download/visualize_detection_annotations.py
    uv run python src/cvmodellearning/download/visualize_detection_annotations.py \
        --annotations runs/<job-id>/data/annotations.json --count 20 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = BACKEND_ROOT / "dataset_cache" / "visionkg"
DEFAULT_OUTPUT = BACKEND_ROOT / "artifacts" / "visionkg_detection_annotations.pdf"
PAGE_SIZE = (1400, 1050)
COLORS = (
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#00c7be", "#ff2d55", "#5856d6", "#a2845e",
)


def _latest_annotations(runs_root: Path) -> Path:
    candidates = [
        path
        for path in runs_root.glob("*/data/annotations.json")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No annotations.json found below {runs_root}. Pass --annotations explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_cache_path(cache_root: Path, relative: str) -> Path:
    candidate = (cache_root / Path(relative.replace("\\", "/"))).resolve()
    root = cache_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Unsafe image_path in annotations: {relative!r}")
    return candidate


def _load_records(annotation_path: Path, cache_root: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in data.get("categories", [])
    }
    by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        by_image[annotation.get("image_id")].append(annotation)

    records = []
    for image_info in data.get("images", []):
        relative = image_info.get("image_path") or image_info.get("file_name")
        if not relative:
            continue
        image_path = _safe_cache_path(cache_root, str(relative))
        annotations = by_image.get(image_info.get("id"), [])
        if image_path.is_file() and annotations:
            records.append({
                "info": image_info,
                "path": image_path,
                "relative": str(relative).replace("\\", "/"),
                "annotations": annotations,
            })
    return records, categories


def _draw_page(record: dict[str, Any], categories: dict[int, str]) -> Image.Image:
    with Image.open(record["path"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    info = record["info"]
    metadata_size = (int(info.get("width", 0)), int(info.get("height", 0)))
    actual_size = image.size
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    header_height = 82
    margin = 24
    available = (PAGE_SIZE[0] - margin * 2, PAGE_SIZE[1] - header_height - margin)
    scale = min(available[0] / image.width, available[1] / image.height)
    shown_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    shown = image.resize(shown_size, Image.Resampling.LANCZOS)
    origin = ((PAGE_SIZE[0] - shown.width) // 2, header_height)
    page.paste(shown, origin)

    mismatch = metadata_size != actual_size
    draw.text((margin, 12), record["relative"], fill="black", font=font)
    draw.text(
        (margin, 34),
        f"actual={actual_size[0]}x{actual_size[1]}  COCO={metadata_size[0]}x{metadata_size[1]}  "
        f"boxes={len(record['annotations'])}" + ("  DIMENSION MISMATCH" if mismatch else ""),
        fill="#c00000" if mismatch else "#333333",
        font=font,
    )
    draw.text((margin, 55), "Boxes interpreted as COCO [x_min, y_min, width, height]", fill="#333333", font=font)

    # Coordinates are defined in the COCO metadata coordinate space. Scaling
    # by actual dimensions intentionally makes metadata mismatches visible.
    sx, sy = shown.width / image.width, shown.height / image.height
    for index, annotation in enumerate(record["annotations"]):
        try:
            x, y, width, height = map(float, annotation["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        x1, y1 = origin[0] + x * sx, origin[1] + y * sy
        x2, y2 = origin[0] + (x + width) * sx, origin[1] + (y + height) * sy
        color = COLORS[int(annotation.get("category_id", index)) % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = categories.get(int(annotation.get("category_id", -1)), "unknown")
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_height = text_box[3] - text_box[1] + 6
        label_y = max(header_height, y1 - text_height)
        draw.rectangle((x1, label_y, x1 + text_box[2] - text_box[0] + 8, label_y + text_height), fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    annotation_path = args.annotations or _latest_annotations(BACKEND_ROOT / "runs")
    records, categories = _load_records(annotation_path.resolve(), args.cache.resolve())
    if not records:
        raise RuntimeError(
            f"No annotated images from {annotation_path} were found in {args.cache}."
        )
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.count, len(records)))
    pages = [_draw_page(record, categories) for record in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.output, "PDF", save_all=True, append_images=pages[1:], resolution=120)
    print(f"Annotations: {annotation_path}")
    print(f"Eligible images: {len(records)}; sampled: {len(selected)}; seed: {args.seed}")
    print(f"PDF: {args.output.resolve()}")


if __name__ == "__main__":
    main()
