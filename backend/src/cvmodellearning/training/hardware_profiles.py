"""Server-controlled hardware profiles used for training and fine-tuning."""

from __future__ import annotations

import os
from typing import Final

from cvmodellearning.schemas.interpretation_schema import TrainingHardwareSpec


DEFAULT_TRAINING_HARDWARE_PROFILE: Final = "macbook_air_m4_16gb"
TRAINING_HARDWARE_PROFILE_ENV: Final = "CVMODEL_TRAINING_HARDWARE_PROFILE"

TRAINING_HARDWARE_PROFILES: Final = {
    "macbook_air_m4_16gb": TrainingHardwareSpec(
        profile_id="macbook_air_m4_16gb",
        accelerator="mps",
        hardware_category="ConsumerGPU",
        gpu_type="Apple M4 10-core GPU",
        gpu_count=1,
        ram_gb=16,
        unified_memory=True,
        training_memory_budget_gb=8,
        max_batch_size=4,
        workers=4,
        supports_amp=False,
    ),
    "rtx6000_48gb": TrainingHardwareSpec(
        profile_id="rtx6000_48gb",
        accelerator="cuda",
        hardware_category="DataCenterGPU",
        gpu_type="NVIDIA RTX 6000 Ada",
        gpu_count=1,
        vram_gb=48,
        unified_memory=False,
        training_memory_budget_gb=44,
        max_batch_size=16,
        workers=8,
        supports_amp=True,
    ),
    "rtx2060_6gb_ryzen5600x_16gb": TrainingHardwareSpec(
        profile_id="rtx2060_6gb_ryzen5600x_16gb",
        accelerator="cuda",
        hardware_category="ConsumerGPU",
        gpu_type="ASUS Dual NVIDIA GeForce RTX 2060 EVO 6GB",
        gpu_count=1,
        vram_gb=6,
        ram_gb=16,
        unified_memory=False,
        # Reserve roughly 1 GB for the display driver, CUDA context, and
        # allocator overhead instead of treating all physical VRAM as usable.
        training_memory_budget_gb=5,
        max_batch_size=4,
        workers=4,
        supports_amp=True,
    ),
}


def get_training_hardware_profile(profile_id: str) -> TrainingHardwareSpec:
    """Return an independent copy of a registered training profile."""
    try:
        return TRAINING_HARDWARE_PROFILES[profile_id].model_copy(deep=True)
    except KeyError as exc:
        available = ", ".join(sorted(TRAINING_HARDWARE_PROFILES))
        raise ValueError(
            f"Unknown training hardware profile {profile_id!r}; available profiles: {available}."
        ) from exc


def active_training_hardware_profile() -> TrainingHardwareSpec:
    """Resolve the server-selected profile for newly created pipeline state."""
    profile_id = os.getenv(
        TRAINING_HARDWARE_PROFILE_ENV,
        DEFAULT_TRAINING_HARDWARE_PROFILE,
    ).strip()
    return get_training_hardware_profile(profile_id)
