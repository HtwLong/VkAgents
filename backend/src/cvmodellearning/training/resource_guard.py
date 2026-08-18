"""Cheap resource checks that run before image tensors are materialized."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from cvmodellearning.models.registry import training_memory_metadata
from cvmodellearning.training.hardware_profiles import active_training_hardware_profile


# This is a corruption/sanity limit, not a model recommendation. Registered
# recipes use substantially smaller inputs, while 4096 still permits specialist
# high-resolution workloads after their model-specific validation.
MAX_IMAGE_SIDE: Final = 4096
GIB: Final = 1024**3


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    supported: bool
    assessment: str
    lower_gb: float | None
    upper_gb: float | None
    budget_gb: float
    model_name: str
    batch_size: int
    image_size: int
    mixed_precision: bool
    components_gb: dict[str, float]
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_training_memory(config: dict) -> TrainingMemoryEstimate:
    """Estimate peak training memory as a conservative range, not an exact value."""

    hardware = active_training_hardware_profile()
    model_name = str(config.get("model_name") or "")
    batch_size = int(config.get("batch_size", 1))
    image_size = int(config.get("image_size", config.get("input_size", 224)))
    mixed = config.get("precision") == "mixed" or config.get("amp") is True
    metadata = training_memory_metadata(model_name)
    if metadata is None or batch_size == -1:
        return TrainingMemoryEstimate(
            supported=False,
            assessment="unverified",
            lower_gb=None,
            upper_gb=None,
            budget_gb=float(hardware.training_memory_budget_gb),
            model_name=model_name,
            batch_size=batch_size,
            image_size=image_size,
            mixed_precision=mixed,
            components_gb={},
            reason=(
                "AutoBatch must be resolved by the backend before analytical estimation."
                if batch_size == -1
                else "No local training-memory metadata exists for this model."
            ),
        )

    parameters = metadata.parameter_count_millions * 1_000_000
    optimizer = str(config.get("optimizer_name") or (config.get("optimizer") or {}).get("name") or "sgd")
    # Includes parameters, gradients, and optimizer state. Mixed precision keeps
    # FP32 master/optimizer copies, so it does not halve this component.
    bytes_per_parameter = 18 if mixed and optimizer in {"adam", "adamw", "rmsprop"} else (
        16 if optimizer in {"adam", "adamw", "rmsprop"} else 12
    )
    parameter_gb = parameters * bytes_per_parameter / GIB
    input_gb = batch_size * 3 * image_size * image_size * (2 if mixed else 4) / GIB
    precision_factor = 0.62 if mixed else 1.0
    activation_gb = (
        batch_size
        * (image_size / 640.0) ** 2
        * metadata.activation_factor
        * 0.35
        * precision_factor
    )
    base = parameter_gb + input_gb + activation_gb + metadata.family_overhead_gb
    lower = base * 0.90
    upper = base * 1.25
    budget = float(hardware.training_memory_budget_gb)
    assessment = "fits" if upper <= budget else "exceeds" if lower > budget else "borderline"
    return TrainingMemoryEstimate(
        supported=True,
        assessment=assessment,
        lower_gb=round(lower, 3),
        upper_gb=round(upper, 3),
        budget_gb=budget,
        model_name=model_name,
        batch_size=batch_size,
        image_size=image_size,
        mixed_precision=mixed,
        components_gb={
            "parameters_gradients_optimizer": round(parameter_gb, 3),
            "activations": round(activation_gb, 3),
            "input_batch": round(input_gb, 3),
            "family_overhead": round(metadata.family_overhead_gb, 3),
        },
    )


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


def validate_training_resource_config(config: dict) -> TrainingMemoryEstimate:
    """Validate resource-sensitive choices before data preparation or training."""

    hardware = active_training_hardware_profile()
    batch_size = int(config.get("batch_size", 1))
    image_size = int(config.get("image_size", config.get("input_size", 224)))
    precision = config.get("precision")
    amp = config.get("amp")

    if batch_size != -1 and batch_size > hardware.max_batch_size:
        raise ValueError(
            f"batch_size={batch_size} exceeds the configured training-hardware maximum "
            f"of {hardware.max_batch_size} for profile '{hardware.profile_id}'. Use gradient "
            "accumulation when a larger effective batch is needed."
        )
    if (precision == "mixed" or amp is True) and not hardware.supports_amp:
        raise ValueError(
            f"Mixed precision is not supported by training-hardware profile "
            f"'{hardware.profile_id}'."
        )
    validate_image_batch_preflight(image_size=image_size, batch_size=batch_size)
    estimate = estimate_training_memory(config)
    if estimate.assessment == "exceeds":
        raise ValueError(
            "Estimated training-memory range "
            f"{estimate.lower_gb:.2f}-{estimate.upper_gb:.2f} GiB exceeds the "
            f"{estimate.budget_gb:.2f} GiB safe budget for model '{estimate.model_name}'. "
            "Reduce batch size or image size, or enable supported mixed precision."
        )
    return estimate


def rank_training_shape_candidates(
    config: dict,
    *,
    image_sizes: tuple[int, ...] = (640, 768, 896, 960),
    batch_sizes: tuple[int, ...] = (4, 2, 1),
) -> list[dict]:
    """Return analytically safe image/batch pairs, highest resolution first.

    These are planning candidates, not measured guarantees. Runtime validation remains
    authoritative and callers must not enable multiscale above the tested image size.
    """

    hardware = active_training_hardware_profile()
    candidates: list[dict] = []
    for image_size in image_sizes:
        for batch_size in batch_sizes:
            if batch_size > hardware.max_batch_size:
                continue
            candidate_config = {
                **config,
                "input_size": image_size,
                "image_size": image_size,
                "batch_size": batch_size,
            }
            estimate = estimate_training_memory(candidate_config)
            if estimate.assessment != "fits":
                continue
            candidates.append({
                "input_size": image_size,
                "batch_size": batch_size,
                "estimated_peak_vram_upper_gb": estimate.upper_gb,
                "training_memory_budget_gb": estimate.budget_gb,
                "validation": "analytical_estimate",
                "requires_measured_runtime_preflight": True,
            })
    return sorted(
        candidates,
        key=lambda item: (-item["input_size"], -item["batch_size"]),
    )
