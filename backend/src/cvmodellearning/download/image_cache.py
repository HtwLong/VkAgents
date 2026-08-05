"""Safe, storage-efficient views of immutable images in the VisionKG cache."""

import errno
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Mapping


def safe_relative_image_path(value: str) -> Path:
    """Validate an image key before using it below a cache or run directory."""
    path = Path(str(value))
    if not str(value).strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe VisionKG image path: {value!r}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Unsafe VisionKG image path: {value!r}")
    return path


def image_path_below(root: Path, value: str) -> Path:
    """Resolve a validated relative image path and enforce root containment."""
    root = root.resolve()
    destination = (root / safe_relative_image_path(value)).resolve(strict=False)
    if not destination.is_relative_to(root):
        raise ValueError(f"VisionKG image path escapes its root: {value!r}")
    return destination


def link_or_copy(source: Path, destination: Path) -> str:
    """Atomically hard-link an immutable file, falling back to symlink then copy."""
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.samefile(source):
        return "existing"

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    method = "hardlink"
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            method = "symlink"
            try:
                temporary.symlink_to(source)
            except OSError:
                method = "copy"
                shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return method
    finally:
        temporary.unlink(missing_ok=True)


def materialize_cached_images(
    images: Iterable[Mapping[str, object]], cache_root: Path, run_root: Path
) -> dict[str, int]:
    """Expose selected cached images at the unchanged paths expected by one run."""
    counts = {"existing": 0, "hardlink": 0, "symlink": 0, "copy": 0}
    for image in images:
        relative_path = str(image.get("image_path") or "")
        source = image_path_below(cache_root, relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Successful download is missing from cache: {source}")
        destination = image_path_below(run_root, relative_path)
        counts[link_or_copy(source, destination)] += 1
    return counts
