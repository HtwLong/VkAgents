"""Safe, storage-efficient views of immutable images in the VisionKG cache."""

import errno
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping


def safe_relative_image_path(value: str) -> Path:
    """Convert a portable image key into a safe native relative path."""
    raw = str(value).strip()
    normalized = raw.replace("\\", "/")
    parts = normalized.split("/")

    unsafe = (
        not raw
        or PurePosixPath(normalized).is_absolute()
        or bool(PureWindowsPath(raw).drive)
        or any(part in {"", ".", ".."} for part in parts)
    )
    if unsafe:
        raise ValueError(f"Unsafe VisionKG image path: {value!r}")
    return Path(*parts)


def image_path_below(root: Path, value: str) -> Path:
    """Return a validated image path below root."""
    return root.resolve() / safe_relative_image_path(value)


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
