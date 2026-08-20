#!/usr/bin/env python3
"""Export presentation-safe run evidence without images or model weights."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MAX_FILE_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".pt", ".pth", ".ckpt", ".safetensors", ".zip", ".tar", ".gz",
}
ALLOWED_ROOT_FILES = {"errors.json", "progress.json", "execution_readiness.json", "run_state.json"}
ALLOWED_ARTIFACT_SUFFIXES = {".json", ".csv", ".txt", ".yaml", ".yml", ".sparql", ".png", ".svg", ".pdf"}
ALLOWED_DATA_FILES = {
    "preparation_summary.json", "dataset_manifest.json",
    "train_labels.csv", "val_labels.csv", "test_labels.csv",
    "train_annotations.json", "val_annotations.json", "test_annotations.json",
    "yolo_data.yaml",
}


def allowed(relative: Path, size: int) -> bool:
    if size > MAX_FILE_BYTES or relative.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if len(relative.parts) == 1:
        return relative.name in ALLOWED_ROOT_FILES
    if relative.parts[0] == "artifacts":
        return relative.suffix.lower() in ALLOWED_ARTIFACT_SUFFIXES
    if relative.parts[0] == "data":
        return relative.name in ALLOWED_DATA_FILES
    return False


def export(source: Path, destination: Path) -> tuple[int, int, int]:
    copied = skipped = total_bytes = 0
    destination.mkdir(parents=True, exist_ok=True)
    for run in sorted(source.iterdir()):
        if not run.is_dir() or run.name.startswith("."):
            continue
        for path in run.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(run)
            size = path.stat().st_size
            if not allowed(relative, size):
                skipped += 1
                continue
            target = destination / run.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
            total_bytes += size
    return copied, skipped, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"source directory not found: {args.source}")
    copied, skipped, total = export(args.source.resolve(), args.destination.resolve())
    print(f"Copied {copied} files ({total / 1024 / 1024:.1f} MiB); skipped {skipped} files.")


if __name__ == "__main__":
    main()

