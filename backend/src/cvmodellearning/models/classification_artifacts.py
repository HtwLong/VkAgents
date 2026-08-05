"""User-facing exports for classification LoRA checkpoints."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from safetensors.torch import save as save_safetensors

from cvmodellearning.models.classification_lora import (
    CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
    LORA_CHECKPOINT_FORMAT,
    validate_classification_lora_metadata,
)
from cvmodellearning.paths import (
    best_model_path,
    lora_adapter_bundle_path,
    merged_model_path,
)


def _load_lora_checkpoint(job_id: str) -> dict:
    path = best_model_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"No classification checkpoint found for job {job_id}.")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_format") != LORA_CHECKPOINT_FORMAT:
        raise ValueError(f"Job {job_id} does not contain a LoRA adapter checkpoint.")
    if checkpoint.get("checkpoint_format_version") != CLASSIFICATION_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Job {job_id} uses an unsupported LoRA checkpoint version.")
    config = checkpoint.get("config", {})
    model_name = str(config.get("model_name", ""))
    if not model_name:
        raise ValueError(f"Job {job_id} has no model_name in its LoRA configuration.")
    validate_classification_lora_metadata(
        model_name,
        config,
        checkpoint.get("adapter_metadata"),
    )
    return checkpoint


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8")


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("peft", "torch", "torchvision", "timm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def ensure_lora_adapter_bundle(job_id: str) -> Path:
    """Create a compact, self-describing adapter download without the base weights."""
    destination = lora_adapter_bundle_path(job_id)
    source = best_model_path(job_id)
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination

    checkpoint = _load_lora_checkpoint(job_id)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in checkpoint["model_state_dict"].items()
    }
    metadata = checkpoint["adapter_metadata"]
    config = checkpoint.get("config", {})
    manifest = {
        "artifact_type": "lora_adapter_bundle",
        "format_version": checkpoint["checkpoint_format_version"],
        "standalone": False,
        "required_base_model": metadata["base_weights_id"],
        "model_name": metadata["model_name"],
        "classes": checkpoint.get("classes", []),
        "package_versions": _package_versions(),
    }
    adapter_config = {
        **metadata,
        "classifier_head_included": True,
    }
    metrics = {
        "best_epoch": checkpoint.get("epoch"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_metric_value": checkpoint.get("best_metric_value"),
    }
    preprocessing = {
        "image_size": config.get("image_size"),
        "resolved": checkpoint.get("resolved_preprocessing", {}),
    }
    readme = (
        "LoRA image-classification adapter\n"
        "=================================\n\n"
        "This bundle is not a standalone model. Reconstruct the exact pretrained base listed in "
        "manifest.json, create the registered classifier head, attach LoRA using adapter_config.json, "
        "then load adapter_model.safetensors. The classifier head is included in that tensor file.\n"
    )

    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("adapter_model.safetensors", save_safetensors(state))
        archive.writestr("adapter_config.json", _json_bytes(adapter_config))
        archive.writestr("manifest.json", _json_bytes(manifest))
        archive.writestr("classes.json", _json_bytes(checkpoint.get("classes", [])))
        archive.writestr("preprocessing.json", _json_bytes(preprocessing))
        archive.writestr("training_config.json", _json_bytes(config))
        archive.writestr("metrics.json", _json_bytes(metrics))
        archive.writestr("README.txt", readme.encode("utf-8"))
    temporary.replace(destination)
    return destination


def ensure_merged_lora_model(job_id: str) -> Path:
    """Create a full state-dict checkpoint with LoRA merged into its pretrained base."""
    destination = merged_model_path(job_id)
    source = best_model_path(job_id)
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination

    checkpoint = _load_lora_checkpoint(job_id)
    config = dict(checkpoint["config"])
    classes = list(checkpoint.get("classes") or config.get("classes") or [])
    from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline

    adapter_model = ClassificationPipeline._restore_checkpoint_model(
        config["model_name"],
        len(classes),
        config,
        checkpoint,
    )
    merged_model = adapter_model.merge_and_unload()
    merged_config = {
        **config,
        "training_mode": "fine_tune_pretrained",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
    export = {
        "epoch": checkpoint.get("epoch"),
        "model_state_dict": merged_model.state_dict(),
        "checkpoint_format": "full_model",
        "checkpoint_format_version": CLASSIFICATION_CHECKPOINT_FORMAT_VERSION,
        "classes": classes,
        "class_to_idx": checkpoint.get("class_to_idx", {}),
        "config": merged_config,
        "resolved_preprocessing": checkpoint.get("resolved_preprocessing", {}),
        "source_adapter_metadata": checkpoint.get("adapter_metadata"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_metric_value": checkpoint.get("best_metric_value"),
    }
    temporary = destination.with_suffix(".tmp")
    torch.save(export, temporary)
    temporary.replace(destination)
    return destination
