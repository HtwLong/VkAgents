"""Downloadable adapter and merged-model exports for RT-DETR LoRA runs."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from safetensors.torch import save as save_safetensors
from ultralytics import RTDETR

from cvmodellearning import paths
from cvmodellearning.models.detection_models.rtdetr_lora import (
    LoRALinear,
    merge_rtdetr_lora_,
    rtdetr_lora_state_dict,
)
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config


RTDETR_LORA_ARTIFACT_VERSION = 1


def _runtime_config(job_id: str) -> dict:
    config_path = paths.hpo_config_path(job_id)
    if not config_path.exists():
        raise FileNotFoundError(f"No hyperparameter configuration found for job {job_id}.")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    return training_compatible_hpo_config(document.get("hyperparameter_candidate") or document)


def _require_lora_job(job_id: str) -> tuple[Path, dict]:
    source = paths.best_yolo_model_path(job_id)
    if not source.exists():
        raise FileNotFoundError(f"No RT-DETR checkpoint found for job {job_id}.")
    config = _runtime_config(job_id)
    if config.get("model_name") != "rtdetr_hgnetv2_l" or config.get("training_mode") != "lora":
        raise ValueError(f"Job {job_id} is not an RT-DETR LoRA run.")
    return source, config


def _versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "ultralytics", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _class_names(model: RTDETR, config: dict) -> list[str]:
    configured = list(config.get("classes") or [])
    if configured:
        return [str(name) for name in configured]
    names = getattr(model.model, "names", {})
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def ensure_rtdetr_lora_adapter_bundle(job_id: str) -> Path:
    """Create a compact bundle containing LoRA tensors and all trained heads."""
    source, config = _require_lora_job(job_id)
    destination = paths.lora_adapter_bundle_path(job_id)
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination

    wrapper = RTDETR(str(source))
    state = rtdetr_lora_state_dict(wrapper.model)
    classes = _class_names(wrapper, config)
    targets = [name for name, module in wrapper.model.named_modules() if isinstance(module, LoRALinear)]
    adapter_config = {
        "artifact_type": "rtdetr_lora_adapter",
        "format_version": RTDETR_LORA_ARTIFACT_VERSION,
        "base_model": "rtdetr_hgnetv2_l",
        "base_checkpoint": "rtdetr-l.pt",
        "rank": int(config.get("lora_rank", 8)),
        "alpha": int(config.get("lora_alpha", 16)),
        "dropout": float(config.get("lora_dropout", 0.05)),
        "target_profile": str(config.get("lora_target_profile", "decoder_attention")),
        "resolved_target_modules": targets,
        "detection_heads_included": True,
    }
    manifest = {
        "artifact_type": "lora_adapter_bundle",
        "format_version": RTDETR_LORA_ARTIFACT_VERSION,
        "standalone": False,
        "required_base_model": "rtdetr-l.pt",
        "model_name": "rtdetr_hgnetv2_l",
        "classes": classes,
        "package_versions": _versions(),
    }
    readme = (
        "RT-DETR LoRA adapter\n"
        "=====================\n\n"
        "This is not a standalone model. It contains LoRA matrices and the fully trained "
        "classification/box heads. Reconstruct the registered rtdetr-l.pt base with the exact "
        "class order in classes.json, inject the profile in adapter_config.json, then load "
        "adapter_model.safetensors. Use best_merged_model.pt for direct Ultralytics inference.\n"
    )
    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("adapter_model.safetensors", save_safetensors(state))
        archive.writestr("adapter_config.json", json.dumps(adapter_config, indent=2).encode())
        archive.writestr("manifest.json", json.dumps(manifest, indent=2).encode())
        archive.writestr("classes.json", json.dumps(classes, indent=2).encode())
        archive.writestr("training_config.json", json.dumps(config, indent=2).encode())
        archive.writestr("README.txt", readme.encode())
    temporary.replace(destination)
    return destination


def ensure_merged_rtdetr_lora_model(job_id: str) -> Path:
    """Create a plain Ultralytics checkpoint with every LoRA update folded in."""
    source, _config = _require_lora_job(job_id)
    destination = paths.merged_detection_model_path(job_id)
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination

    wrapper = RTDETR(str(source))
    if not any(isinstance(module, LoRALinear) for module in wrapper.model.modules()):
        raise ValueError("RT-DETR training checkpoint contains no LoRA layers to merge.")
    merge_rtdetr_lora_(wrapper.model)
    if any(isinstance(module, LoRALinear) for module in wrapper.model.modules()):
        raise ValueError("RT-DETR LoRA merge left custom adapter modules in the model.")
    # Ultralytics prefers ckpt['ema'] over ckpt['model'] while loading. It is
    # stale and still contains LoRALinear modules, so ensure save() cannot retain it.
    wrapper.ckpt["ema"] = None
    wrapper.ckpt["optimizer"] = None
    wrapper.ckpt["updates"] = None
    temporary = destination.with_suffix(".tmp")
    wrapper.save(temporary)
    temporary.replace(destination)

    # Validate the actual user-facing load path, not only the in-memory merge.
    restored = RTDETR(str(destination))
    if any(isinstance(module, LoRALinear) for module in restored.model.modules()):
        destination.unlink(missing_ok=True)
        raise ValueError("Merged RT-DETR artifact still depends on custom LoRA modules.")
    return destination
