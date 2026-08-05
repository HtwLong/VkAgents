from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GIB_BYTES = 1024**3
DEFAULT_RUNTIME_OVERHEAD_FRACTION = 0.20
PRECISION_BYTES = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "INT8": 1.0,
    "INT4": 0.5,
}

CNNActivationTask = Literal["classification", "detection"]


@dataclass(frozen=True)
class InferenceMemoryEstimate:
    """Analytical model-memory components in GiB.

    Activation workspace is an explicit input because it cannot be derived
    reliably from parameter count or FLOPs alone.
    """

    weight_memory_gb: float
    activation_workspace_gb: float
    kv_cache_gb: float
    runtime_overhead_gb: float
    total_estimated_vram_gb: float


def _rounded_gib(value: float) -> float:
    return round(float(value), 3)


def calculate_inference_memory(
    *,
    params_m: float,
    precision_mode: str,
    activation_workspace_gb: float,
    kv_cache_gb: float = 0.0,
    runtime_overhead_fraction: float = DEFAULT_RUNTIME_OVERHEAD_FRACTION,
) -> InferenceMemoryEstimate:
    """Calculate the reproducible components used by the ontology estimates.

    Parameter count and precision determine weight memory. The default runtime
    overhead is 20% of unrounded weight memory, following the rule of thumb
    cited by the ontology's Hugging Face Accelerate evidence. Components are
    rounded before summation to reproduce the stored CSV convention.
    """
    if params_m <= 0:
        raise ValueError("params_m must be greater than zero.")
    if precision_mode not in PRECISION_BYTES:
        supported = ", ".join(PRECISION_BYTES)
        raise ValueError(f"Unsupported precision_mode '{precision_mode}'; use one of: {supported}.")
    if activation_workspace_gb < 0 or kv_cache_gb < 0:
        raise ValueError("Activation workspace and KV-cache memory cannot be negative.")
    if runtime_overhead_fraction < 0:
        raise ValueError("runtime_overhead_fraction cannot be negative.")

    weight_memory = params_m * 1_000_000 * PRECISION_BYTES[precision_mode] / GIB_BYTES
    weight_memory_gb = _rounded_gib(weight_memory)
    activation_gb = _rounded_gib(activation_workspace_gb)
    cache_gb = _rounded_gib(kv_cache_gb)
    overhead_gb = _rounded_gib(weight_memory * runtime_overhead_fraction)
    total_gb = _rounded_gib(weight_memory_gb + activation_gb + cache_gb + overhead_gb)

    return InferenceMemoryEstimate(
        weight_memory_gb=weight_memory_gb,
        activation_workspace_gb=activation_gb,
        kv_cache_gb=cache_gb,
        runtime_overhead_gb=overhead_gb,
        total_estimated_vram_gb=total_gb,
    )


def estimate_cnn_activation_workspace(
    *,
    flops_b: float,
    task: CNNActivationTask,
    batch_size: int = 1,
    precision_mode: str = "FP16",
) -> float:
    """Apply the ontology's explicit local CNN activation heuristic.

    This is a consistency policy, not a source-measured memory value. Existing
    classification rows use max(0.05, 0.01 * GFLOPs) GiB and detection rows use
    max(0.25, 0.005 * GFLOPs) GiB for FP16 batch size one. Batch and dtype scaling
    are treated as approximately linear and should be replaced by measurement
    for deployment.
    """
    if flops_b < 0:
        raise ValueError("flops_b cannot be negative.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    if precision_mode not in PRECISION_BYTES:
        supported = ", ".join(PRECISION_BYTES)
        raise ValueError(f"Unsupported precision_mode '{precision_mode}'; use one of: {supported}.")

    floor_gb, gib_per_gflop = {
        "classification": (0.05, 0.01),
        "detection": (0.25, 0.005),
    }[task]
    precision_scale = PRECISION_BYTES[precision_mode] / PRECISION_BYTES["FP16"]
    return _rounded_gib(
        max(floor_gb, gib_per_gflop * flops_b) * batch_size * precision_scale
    )


def calculate_kv_cache_memory(
    *,
    batch_size: int,
    context_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype_bytes: float = 2.0,
) -> float:
    """Calculate autoregressive key/value-cache memory in GiB."""
    values = (batch_size, context_tokens, layers, kv_heads, head_dim)
    if any(value < 1 for value in values):
        raise ValueError("KV-cache dimensions must all be at least one.")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be greater than zero.")

    cache_bytes = (
        batch_size
        * context_tokens
        * layers
        * kv_heads
        * head_dim
        * 2  # key and value tensors
        * dtype_bytes
    )
    return _rounded_gib(cache_bytes / GIB_BYTES)
