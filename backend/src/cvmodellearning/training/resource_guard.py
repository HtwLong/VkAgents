"""Cheap resource checks that run before image tensors are materialized."""

from __future__ import annotations

from typing import Final

from cvmodellearning.training.hardware_profiles import active_training_hardware_profile


# This is a corruption/sanity limit, not a model recommendation. Registered
# recipes use substantially smaller inputs, while 4096 still permits specialist
# high-resolution workloads after their model-specific validation.
MAX_IMAGE_SIDE: Final = 4096


def validate_image_batch_preflight(
    *,
    image_size: int,
    batch_size: int,
    channels: int = 3,
    bytes_per_element: int = 4,
) -> None:
    """Reject an impossible raw image batch before transforms allocate it."""
    if image_size > MAX_IMAGE_SIDE:
        raise ValueError(
            f"image_size/input_size must be <= {MAX_IMAGE_SIDE}; received {image_size}. "
            "This safety limit prevents malformed configurations from allocating enormous tensors."
        )
    if batch_size < 1:
        # Detection's -1 AutoBatch is resolved by its backend.
        return

    required_bytes = batch_size * channels * image_size * image_size * bytes_per_element
    hardware = active_training_hardware_profile()
    budget_bytes = int(hardware.training_memory_budget_gb * 1024**3)
    if required_bytes > budget_bytes:
        required_gib = required_bytes / 1024**3
        raise ValueError(
            f"Raw input batch requires {required_gib:.2f} GiB, exceeding the "
            f"{hardware.training_memory_budget_gb:g} GiB training-memory budget for "
            f"hardware profile '{hardware.profile_id}'. Reduce image size or batch size."
        )
